"""
AI Pipeline service for legal document analysis.

This module implements the core AI functionality using LangChain to analyze
legal and policy documents, extract entities, and generate summaries.
"""

import json
import time
import asyncio
from typing import List, Optional
from uuid import UUID
import re

from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangChainDocument
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from src.core.config import settings
from src.core.logging import LoggingMixin, log_ai_operation, debug_log
from src.core.models import (
    AnalysisResult,
    ConfidenceLevel,
    Document,
    DocumentSummary,
    ExtractedEntity,
    KeyProvision,
    RiskAssessment,
)
from src.core.repository import AnalysisRepository


class AIAnalysisError(Exception):
    """Custom exception for AI analysis errors."""

    pass


class AIAnalysisPipeline(LoggingMixin):
    """AI pipeline for analyzing legal and policy documents."""

    def __init__(self, analysis_repository: Optional[AnalysisRepository] = None):
        """Initialize the AI analysis pipeline.

        Args:
            analysis_repository: Repository for saving analysis results.
                                If None, creates a default instance.
        """
        self.llm = self._initialize_llm()
        self.confidence_threshold = settings.confidence_threshold

        # Use dependency injection for repository
        if analysis_repository is None:
            from src.core.repository import analysis_repository as default_repo

            self.analysis_repository = default_repo
        else:
            self.analysis_repository = analysis_repository

    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize the language model based on configuration.

        Returns:
            Initialized language model instance.

        Raises:
            AIAnalysisError: If LLM initialization fails.
        """
        try:
            llm_config = settings.get_llm_config()

            # Check if we should use a mock LLM for testing
            if llm_config.get("mock", False) or (
                llm_config.get("api_key", "").startswith(("sk-test-", "sk-ant-test-"))
            ):
                # Return a mock LLM for testing
                from src.pipeline.mock_llm import MockLLM

                debug_log("Using MockLLM for testing")
                return MockLLM()

            if llm_config["provider"] == "openai":
                debug_log("Initializing ChatOpenAI", model=llm_config["model"])
                return ChatOpenAI(
                    model=llm_config["model"],
                    api_key=llm_config["api_key"],
                    temperature=0.1,  # Low temperature for more consistent outputs
                    max_tokens=2000,
                )
            elif llm_config["provider"] == "ollama":
                debug_log("Initializing ChatOllama", model=llm_config["model"])
                return ChatOllama(
                    model=llm_config["model"],
                    api_key=llm_config["api_key"],
                    temperature=0.1,  # Low temperature for more consistent outputs
                    max_tokens=2000,
                )
            else:
                # Future: Add support for other providers
                raise AIAnalysisError(
                    f"Unsupported LLM provider: {llm_config['provider']}"
                )

        except Exception as e:
            raise AIAnalysisError(f"Failed to initialize LLM: {str(e)}")

    async def analyze_document(self, document: Document) -> AnalysisResult:
        """Perform comprehensive analysis of a legal document.

        Args:
            document: The document to analyze.

        Returns:
            AnalysisResult: Complete analysis results.

        Raises:
            AIAnalysisError: If analysis fails.
        """
        start_time = time.time()

        try:
            # Extract entities
            entities = await self._extract_entities(document)

            self.logger.info(
                "Extracted entities",
                document_id=str(document.id),
                filename=document.filename,
                entitiesCount=len(entities),
            )

            # Generate summary with key provisions and risk assessments
            summary = await self._generate_summary(document, entities)

            # Determine overall confidence level
            confidence_level = self._calculate_confidence_level(
                summary.confidence_score
            )

            # Check if human review is needed
            requires_review = self._requires_human_review(summary, entities)

            # Generate warnings
            warnings = self._generate_warnings(summary, entities)

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            result = AnalysisResult(
                document_id=document.id,
                entities=entities,
                summary=summary,
                confidence_level=confidence_level,
                processing_time_ms=processing_time,
                model_used=(
                    self.llm.model_name
                    if hasattr(self.llm, "model_name")
                    else "unknown"
                ),
                warnings=warnings,
                requires_human_review=requires_review,
            )

            log_ai_operation(
                operation="document_analysis",
                model=result.model_used,
                confidence=summary.confidence_score,
                duration_ms=processing_time,
            )

            # Persist results to the database using injected repository
            try:
                self.analysis_repository.create(result)
                self.logger.info(
                    "Analysis result persisted to database",
                    analysis_id=str(result.id),
                    document_id=str(result.document_id),
                )
            except Exception as e:
                self.logger.error(
                    "Failed to persist analysis result",
                    analysis_id=str(result.id),
                    error=str(e),
                    exc_info=True,
                )
                # Don't fail the entire analysis if persistence fails

            return result

        except Exception as e:
            self.logger.error(
                "Document analysis failed",
                document_id=str(document.id),
                error=str(e),
                exc_info=True,
            )
            raise AIAnalysisError(f"Analysis failed: {str(e)}")

    async def _extract_entities(self, document: Document) -> List[ExtractedEntity]:
        """Extract legal entities from the document.

        Args:
            document: The document to extract entities from.

        Returns:
            List of extracted entities.
        """
        entity_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following legal/policy document and extract key entities.
            For each entity, provide:
            1. Entity type (law, agency, affected_group, legal_concept, jurisdiction, etc.)
            2. Entity value (the actual name/reference)
            3. Confidence score (0.0-1.0)
            4. Source text (the exact text where found)
            5. Position information (approximate character positions)
            
            Document text:
            {text}
            
            Return the results in the following format for each entity:
            Type: [entity_type]
            Value: [entity_value]
            Confidence: [confidence_score]
            Source: [source_text]
            Start: [start_position]
            End: [end_position]
            ---
            """,
        )

        try:
            # Split document into chunks if too long
            text_chunks = self._split_text(document.content)
            all_entities = []

            for chunk_idx, chunk in enumerate(text_chunks):
                prompt = entity_prompt.format(text=chunk)
                response = await self.llm.ainvoke(prompt)

                # Parse the response and create ExtractedEntity objects
                entities = self._parse_entity_response(
                    response.content, chunk_idx * 1000
                )
                all_entities.extend(entities)

            return all_entities

        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e))
            # Return empty list instead of failing completely
            return []

    async def _generate_summary(
        self, document: Document, entities: List[ExtractedEntity]
    ) -> DocumentSummary:
        """Generate a comprehensive summary of the document.

        Args:
            document: The document to summarize.
            entities: Previously extracted entities.

        Returns:
            DocumentSummary: Comprehensive document summary.
        """
        try:
            # Split document into chunks
            text_chunks = self._split_text(document.content)

            # Concurrently summarize all chunks
            summarize_tasks = [self._summarize_chunk(chunk) for chunk in text_chunks]
            chunk_summaries = await asyncio.gather(*summarize_tasks)

            # Combine individual chunk summaries into a final comprehensive summary
            summary = await self._combine_summaries(chunk_summaries, entities)

            self.logger.debug("** summary parsed **", summary=summary)
            return summary

        except Exception as e:
            self.logger.error("Summary generation failed", error=str(e))
            # Return a basic summary instead of failing
            return DocumentSummary(
                executive_summary="Summary generation failed. Manual review required.",
                key_points=["Analysis incomplete due to processing error"],
                main_provisions=[],
                risk_assessments=[],
                affected_groups=[],
                legal_precedents=[],
                confidence_score=0.1,
            )

    async def _summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk of text.

        Args:
            chunk: A chunk of the document text.

        Returns:
            The summary of the chunk.
        """
        prompt_template = PromptTemplate(
            input_variables=["chunk_text"],
            template="""
            Please summarize the following text from a legal document.
            Focus on extracting the key facts, arguments, and conclusions.

            Text:
            {chunk_text}

            Summary:
            """,
        )
        prompt = prompt_template.format(chunk_text=chunk)
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def _combine_summaries(
        self, summaries: List[str], entities: List[ExtractedEntity]
    ) -> DocumentSummary:
        """Combine individual chunk summaries into a final comprehensive summary.

        Args:
            summaries: A list of summaries from each document chunk.
            entities: A list of extracted entities from the document.

        Returns:
            The final, consolidated DocumentSummary.
        """
        combined_summary_text = "\n".join(summaries)
        entities_text = "\n".join(
            [
                f"- {e.entity_type}: {e.entity_value} (confidence: {e.confidence})"
                for e in entities
            ]
        )

        final_prompt_template = PromptTemplate(
            input_variables=["combined_summaries", "entities"],
            template="""
            You are an expert legal analyst. The following are summaries of sections from a single legal or policy document.
            Please synthesize these summaries into a single, cohesive, and comprehensive final analysis.

            Section Summaries:
            {combined_summaries}

            Extracted Entities:
            {entities}

            Please provide:
            1. Executive Summary (2-3 paragraphs, capturing the essence of the document)
            2. Key Points (bullet list of the most critical takeaways)
            3. Main Provisions (detailed explanation of the document's main articles or sections and their impact)
            4. Risk Assessments (potential risks related to civil rights, privacy, constitutional issues, etc.)
            5. Affected Groups (populations or sectors most impacted by this document)
            6. Legal Precedents/References (any mentioned legal cases, statutes, or regulations)
            7. Implementation Timeline (if specified in the document)
            8. Overall Confidence Score (a float from 0.0 to 1.0, assessing the quality and completeness of your own summary based *only* on the provided text)

            Format your response clearly with markdown headers for each section.
            Ensure the final output is well-structured and easy to read.
            """,
        )

        prompt = final_prompt_template.format(
            combined_summaries=combined_summary_text, entities=entities_text
        )
        self.logger.debug("** Sending final combination prompt to llm **")
        response = await self.llm.ainvoke(prompt)
        self.logger.debug("** Final combination response received **")

        return self._parse_summary_response(response.content)

    def _split_text(
        self, text: str, chunk_size: int = 3000, chunk_overlap: int = 200
    ) -> List[str]:
        """Split text into manageable chunks for processing.

        Args:
            text: Text to split.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    def _parse_entity_response(
        self, response: str, offset: int = 0
    ) -> List[ExtractedEntity]:
        """Parse the LLM response to extract entities.

        Args:
            response: Raw LLM response.
            offset: Character offset for position calculation.

        Returns:
            List of parsed entities.
        """

        debug_log(
            "** Parsing entity response:",
            response=(response[:100] + "...") if len(response) > 100 else response,
        )

        # probably need to improve prompt to get the format consistent between models
        # but for now let's try to account for variations we have seen
        if re.search(r"```json", response):
            json_response = response.split("```json")[1].split("```")[0]
            response_entities = json.loads(json_response)

            entities = [self._map_to_entity(entity) for entity in response_entities]
            return entities

        # sometimes it's wrapped in a markdown code string for no apparent reason
        if re.search(r"```$", response):
            response = response.split("```")[1]

        entities = []

        sections = response.split("---")

        for section in sections:
            if not section.strip():
                continue

            try:
                lines = [
                    line.strip() for line in section.strip().split("\n") if line.strip()
                ]
                if len(lines) < 6:
                    continue

                match = re.search(r"\*\*Type:\*\*", lines[0])
                if match:
                    entity_type = re.sub(r"\*\*Type:\*\*", "", lines[0]).strip()
                    entity_value = re.sub(r"\*\*Value:\*\*", "", lines[1]).strip()
                    confidence = float(
                        re.sub(r"\*\*Confidence:\*\*", "", lines[2]).strip()
                    )
                    source_text = re.sub(r"\*\*Source:\*\*", "", lines[3]).strip()
                    start_pos = (
                        int(re.sub(r"\*\*Start:\*\*", "", lines[4]).strip()) + offset
                    )
                    end_pos = (
                        int(re.sub(r"\*\*End:\*\*", "", lines[5]).strip()) + offset
                    )
                else:
                    entity_type = re.sub(r"Type:", "", lines[0]).strip()
                    entity_value = re.sub(r"Value:", "", lines[1]).strip()
                    confidence = float(re.sub(r"Confidence:", "", lines[2]).strip())
                    source_text = re.sub(r"Source:", "", lines[3]).strip()
                    start_pos = int(re.sub(r"Start:", "", lines[4]).strip()) + offset
                    end_pos = int(re.sub(r"End:", "", lines[5]).strip()) + offset

                entity = ExtractedEntity(
                    entity_type=entity_type,
                    entity_value=entity_value,
                    confidence=confidence,
                    source_text=source_text,
                    start_position=start_pos,
                    end_position=end_pos,
                )
                entities.append(entity)

            except (ValueError, IndexError) as e:
                self.logger.warning("Failed to parse entity", error=str(e))
                debug_log("lines that it failed to parse", lines)
                continue

        return entities

    def _map_to_entity(self, entity: dict) -> ExtractedEntity:
        """Map the entity to the ExtractedEntity model."""
        try:
            start_pos = entity.get("Start")
            if start_pos is None:
                start_pos = entity.get("start_position")

            end_pos = entity.get("End")
            if end_pos is None:
                end_pos = entity.get("end_position")

            return ExtractedEntity(
                entity_type=entity.get("Type") or entity.get("entity_type"),
                entity_value=entity.get("Value") or entity.get("entity_value"),
                confidence=entity.get("Confidence") or entity.get("confidence"),
                source_text=entity.get("Source") or entity.get("source_text"),
                start_position=start_pos,
                end_position=end_pos,
            )
        except Exception as e:
            self.logger.error(f"Failed to map entity: {str(entity)}", error=str(e))
            return ExtractedEntity(
                entity_type="unknown",
                entity_value="unknown",
                confidence=0.0,
                source_text="unknown",
                start_position=0,
                end_position=0,
            )

    def _parse_summary_response(self, response: str) -> DocumentSummary:
        """Parse the LLM response to create a structured summary.

        Args:
            response: Raw LLM response.

        Returns:
            Parsed DocumentSummary.
        """
        # Simple parsing - in production, use more robust parsing or structured output

        # Default values
        executive_summary = "Summary not available"
        key_points = []
        main_provisions = []
        risk_assessments = []
        affected_groups = []
        legal_precedents = []
        confidence_score = 0.5

        try:
            # Extract executive summary (first paragraph)
            lines = response.split("\n")
            summary_lines = []
            for line in lines[1:6]:  # Skip first line, take next 5
                if line.strip():
                    summary_lines.append(line.strip())
            executive_summary = (
                " ".join(summary_lines) if summary_lines else executive_summary
            )

            # Extract confidence score (look for pattern)
            for line in lines:
                if "confidence" in line.lower() and any(
                    char.isdigit() for char in line
                ):
                    try:
                        # Extract number between 0 and 1
                        import re

                        match = re.search(r"(\d*\.?\d+)", line)
                        if match:
                            score = float(match.group(1))
                            if score <= 1.0:
                                confidence_score = score
                            elif score <= 100:
                                confidence_score = score / 100
                    except ValueError:
                        pass

        except Exception as e:
            self.logger.warning("Failed to parse summary response", error=str(e))

        return DocumentSummary(
            executive_summary=executive_summary,
            key_points=key_points,
            main_provisions=main_provisions,
            risk_assessments=risk_assessments,
            affected_groups=affected_groups,
            legal_precedents=legal_precedents,
            confidence_score=confidence_score,
        )

    def _calculate_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Calculate overall confidence level from score.

        Args:
            confidence_score: Numerical confidence score.

        Returns:
            ConfidenceLevel: Categorical confidence level.
        """
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _requires_human_review(
        self, summary: DocumentSummary, entities: List[ExtractedEntity]
    ) -> bool:
        """Determine if the analysis requires human review.

        Args:
            summary: Generated summary.
            entities: Extracted entities.

        Returns:
            bool: True if human review is recommended.
        """
        # Low confidence
        if summary.confidence_score < self.confidence_threshold:
            return True

        # High-risk assessments
        high_risk_count = sum(
            1 for risk in summary.risk_assessments if risk.risk_level.lower() == "high"
        )
        if high_risk_count > 2:
            return True

        # Low entity confidence
        low_confidence_entities = sum(
            1 for entity in entities if entity.confidence < 0.6
        )
        if (
            low_confidence_entities > len(entities) * 0.3
        ):  # More than 30% low confidence
            return True

        return False

    def _generate_warnings(
        self, summary: DocumentSummary, entities: List[ExtractedEntity]
    ) -> List[str]:
        """Generate warnings based on analysis results.

        Args:
            summary: Generated summary.
            entities: Extracted entities.

        Returns:
            List of warning messages.
        """
        warnings = []

        if summary.confidence_score < 0.5:
            warnings.append(
                "⚠️ Low confidence analysis. Please review the original document."
            )

        if not entities:
            warnings.append(
                "⚠️ No entities were extracted. Document may require manual review."
            )

        if len(summary.risk_assessments) == 0:
            warnings.append(
                "⚠️ No risk assessments generated. Consider manual risk evaluation."
            )

        return warnings
