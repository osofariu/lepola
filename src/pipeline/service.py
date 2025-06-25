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
            self.logger.info("LLM config", llm_config=llm_config)

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
                    request_timeout=settings.http_timeout,
                )
            elif llm_config["provider"] == "ollama":
                debug_log("Initializing ChatOllama", model=llm_config["model"])
                return ChatOllama(
                    model=llm_config["model"],
                    api_key=llm_config["api_key"],
                    temperature=0.05,  # Lower temperature for more consistent outputs
                    max_tokens=4000,  # Increased for more detailed analysis
                    base_url=settings.ollama_base_url,
                    # Note: ChatOllama doesn't have a direct timeout parameter
                    # The timeout is handled by the underlying HTTP client
                )
            else:
                # Future: Add support for other providers
                raise AIAnalysisError(
                    f"Unsupported LLM provider: {llm_config['provider']}"
                )

        except Exception as e:
            raise AIAnalysisError(f"Failed to initialize LLM: {str(e)}")

    async def analyze_document(
        self, document: Document, force_regenerate_entities: bool = False
    ) -> AnalysisResult:
        """Perform comprehensive analysis of a legal document.

        Args:
            document: The document to analyze.
            force_regenerate_entities: If True, regenerate entities even if they exist.

        Returns:
            AnalysisResult: Complete analysis results.

        Raises:
            AIAnalysisError: If analysis fails.
        """
        start_time = time.time()

        try:
            # Check for existing entities unless forced to regenerate
            entities = []
            entities_source_analysis_id = None

            if not force_regenerate_entities:
                # Try to find existing entities from a previous analysis
                existing_analysis = (
                    self.analysis_repository.get_latest_analysis_with_entities(
                        document.id
                    )
                )
                if existing_analysis and existing_analysis.entities:
                    entities = existing_analysis.entities
                    entities_source_analysis_id = existing_analysis.id
                    self.logger.info(
                        "Using cached entities from previous analysis",
                        document_id=str(document.id),
                        filename=document.filename,
                        entitiesCount=len(entities),
                        source_analysis_id=str(entities_source_analysis_id),
                    )

            # Extract entities if we don't have them from cache
            if not entities:
                entities = await self._extract_entities(document)
                self.logger.info(
                    "Extracted new entities",
                    document_id=str(document.id),
                    filename=document.filename,
                    entitiesCount=len(entities),
                )

            # Generate summary with key provisions and risk assessments
            summary = await self._generate_summary(document, entities)

            # Calculate comprehensive confidence level
            comprehensive_confidence = self._calculate_comprehensive_confidence(
                summary, entities
            )
            confidence_level = self._calculate_confidence_level(
                comprehensive_confidence
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
                    self.llm.model if hasattr(self.llm, "model") else "unknown"
                ),
                warnings=warnings,
                requires_human_review=requires_review,
                entities_source_analysis_id=entities_source_analysis_id,
            )

            log_ai_operation(
                operation="document_analysis",
                model=result.model_used,
                confidence=summary.confidence_score,
                duration_ms=processing_time,
            )

            # Persist results to the database using injected repository with retry logic
            await self._persist_analysis_result_with_retry(result)

            return result

        except Exception as e:
            self.logger.error(
                "Document analysis failed",
                document_id=str(document.id),
                error=str(e),
                exc_info=True,
            )
            raise AIAnalysisError(f"Analysis failed: {str(e)}")

    async def _persist_analysis_result_with_retry(
        self, result: AnalysisResult, max_retries: int = 3
    ) -> None:
        """Persist analysis result to database with retry logic.

        Args:
            result: Analysis result to persist.
            max_retries: Maximum number of retry attempts.
        """
        for attempt in range(max_retries):
            try:
                self.analysis_repository.create(result)
                self.logger.info(
                    "Analysis result persisted to database",
                    analysis_id=str(result.id),
                    document_id=str(result.document_id),
                    attempt=attempt + 1,
                )
                return
            except Exception as e:
                self.logger.warning(
                    "Failed to persist analysis result (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                    analysis_id=str(result.id),
                    error=str(e),
                )

                if attempt == max_retries - 1:
                    # Last attempt failed, log error but don't fail the entire analysis
                    self.logger.error(
                        "Failed to persist analysis result after %d attempts",
                        max_retries,
                        analysis_id=str(result.id),
                        error=str(e),
                        exc_info=True,
                    )
                    # Don't raise the exception - the analysis was successful,
                    # we just couldn't persist it
                else:
                    # Wait before retrying (exponential backoff)
                    await asyncio.sleep(2**attempt)

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
            You are an expert legal analyst with deep knowledge of legal documents, regulations, and policy analysis.
            
            Analyze the following legal/policy document and extract key entities with high precision.
            
            INSTRUCTIONS:
            1. Read the text carefully and identify all significant legal entities
            2. For each entity, provide a confidence score based on:
               - Clarity of the entity in the text (0.9-1.0 for explicit mentions)
               - Ambiguity level (0.7-0.9 for clear but contextual mentions)
               - Inference required (0.5-0.7 for implied entities)
            3. Be conservative with confidence scores - only assign high confidence when certain
            4. Focus on legal relevance and significance
            
            Entity types to look for:
            - legal_document: Laws, regulations, statutes, bills
            - agency: Government agencies, departments, organizations
            - affected_group: Populations, demographics, stakeholders
            - legal_concept: Rights, obligations, procedures, standards
            - jurisdiction: Geographic or legal jurisdictions
            - timeline: Dates, deadlines, effective dates
            - penalty: Fines, sanctions, enforcement mechanisms
            
            Document text:
            {text}
            
            Return the results in the following JSON format:
            ```json
            [
              {{
                "Type": "entity_type",
                "Value": "entity_value",
                "Confidence": confidence_score,
                "Source": "exact_source_text",
                "Start": start_position,
                "End": end_position
              }}
            ]
            ```
            
            IMPORTANT: Only include entities you are confident about. Quality over quantity.
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
            You are an expert legal analyst with decades of experience in policy analysis, constitutional law, and regulatory compliance.
            
            Your task is to provide a comprehensive, accurate, and well-reasoned analysis of a legal or policy document.
            
            Section Summaries:
            {combined_summaries}

            Extracted Entities:
            {entities}

            ANALYSIS REQUIREMENTS:
            
            1. **Executive Summary** (2-3 paragraphs)
               - Capture the document's essence and primary purpose
               - Highlight the most significant legal implications
               - Be precise and avoid speculation
            
            2. **Key Points** (bullet list)
               - Focus on actionable insights and critical takeaways
               - Prioritize legal significance and practical impact
               - Include compliance requirements and deadlines
            
            3. **Main Provisions** (detailed analysis)
               - Analyze each major section for legal authority and scope
               - Assess implementation requirements and enforcement mechanisms
               - Evaluate potential challenges and constitutional considerations
            
            4. **Risk Assessments** (comprehensive evaluation)
               - Civil rights implications and potential violations
               - Privacy concerns and data protection issues
               - Constitutional challenges and legal precedents
               - Economic impact and stakeholder effects
            
            5. **Affected Groups** (stakeholder analysis)
               - Directly impacted populations and organizations
               - Secondary effects on related sectors
               - Compliance burden assessment
            
            6. **Legal Precedents/References**
               - Cited legal authorities and their relevance
               - Related case law and regulatory frameworks
               - Potential conflicts with existing law
            
            7. **Implementation Timeline**
               - Effective dates and phase-in periods
               - Agency rulemaking requirements
               - Compliance deadlines and milestones
            
            8. **Confidence Assessment**
               - Evaluate the quality and completeness of your analysis
               - Consider: clarity of source material, comprehensiveness of coverage, certainty of legal interpretations
               - Provide a confidence score (0.0-1.0) with justification
               - Be conservative - only assign high confidence when analysis is thorough and source material is clear
            
            FORMAT: Use clear markdown headers and structured formatting.
            
            CONFIDENCE SCORING GUIDELINES:
            - 0.9-1.0: Clear, comprehensive analysis with unambiguous source material
            - 0.7-0.8: Good analysis with minor uncertainties or gaps
            - 0.5-0.6: Adequate analysis with significant limitations
            - 0.3-0.4: Limited analysis due to unclear or incomplete source material
            - 0.1-0.2: Minimal analysis possible due to poor source quality
            
            Provide a thorough, professional analysis that would be suitable for legal counsel and policy makers.
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
        self, text: str, chunk_size: int = 4000, chunk_overlap: int = 500
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

            # Try to break at sentence boundaries to preserve context
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + chunk_size - 200, start)
                search_end = min(end + 200, len(text))

                # Find the last sentence boundary
                last_period = text.rfind(".", search_start, search_end)
                last_exclamation = text.rfind("!", search_start, search_end)
                last_question = text.rfind("?", search_start, search_end)

                # Use the latest sentence boundary
                sentence_end = max(last_period, last_exclamation, last_question)

                if (
                    sentence_end > start + chunk_size * 0.8
                ):  # Only use if it's not too early
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break

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
        if confidence_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.65:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _calculate_comprehensive_confidence(
        self, summary: DocumentSummary, entities: List[ExtractedEntity]
    ) -> float:
        """Calculate a comprehensive confidence score based on multiple factors.

        Args:
            summary: Generated summary.
            entities: Extracted entities.

        Returns:
            float: Comprehensive confidence score (0.0-1.0).
        """
        # Base confidence from summary
        base_confidence = summary.confidence_score

        # Entity confidence factor
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            high_confidence_entities = sum(1 for e in entities if e.confidence >= 0.8)
            entity_quality_ratio = high_confidence_entities / len(entities)
        else:
            avg_entity_confidence = 0.0
            entity_quality_ratio = 0.0

        # Content quality factors
        summary_length = len(summary.executive_summary)
        has_key_points = len(summary.key_points) > 0
        has_provisions = len(summary.main_provisions) > 0
        has_risks = len(summary.risk_assessments) > 0

        # Calculate content completeness score
        content_factors = [
            1.0 if summary_length > 200 else 0.5,  # Good summary length
            1.0 if has_key_points else 0.3,  # Has key points
            1.0 if has_provisions else 0.3,  # Has provisions
            1.0 if has_risks else 0.3,  # Has risk assessments
        ]
        content_completeness = sum(content_factors) / len(content_factors)

        # Weighted confidence calculation
        weights = {"base": 0.4, "entities": 0.3, "entity_quality": 0.2, "content": 0.1}

        comprehensive_confidence = (
            base_confidence * weights["base"]
            + avg_entity_confidence * weights["entities"]
            + entity_quality_ratio * weights["entity_quality"]
            + content_completeness * weights["content"]
        )

        return min(1.0, max(0.0, comprehensive_confidence))

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
