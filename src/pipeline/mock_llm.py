"""
Mock LLM for testing purposes.

This module provides a mock implementation of the LangChain BaseLanguageModel
interface for testing and development when real API keys are not available.
"""

from typing import Any, List, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult, Generation


class MockLLM(BaseLanguageModel):
    """Mock LLM implementation for testing.

    This class provides realistic mock responses for legal document analysis
    without requiring real API keys or making external API calls.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the mock LLM."""
        super().__init__(**kwargs)
        # Set model_name using __dict__ to bypass Pydantic validation
        object.__setattr__(self, "model_name", "mock-gpt-4")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate mock responses based on the input."""
        # Get the last message content
        last_message = messages[-1].content if messages else ""

        # Generate appropriate mock response based on prompt content
        mock_response = self._generate_mock_response(last_message)

        generation = Generation(text=mock_response)
        return LLMResult(generations=[[generation]])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async version of generate."""
        return self._generate(messages, stop, run_manager, **kwargs)

    def invoke(
        self, input: Any, config: Optional[Any] = None, **kwargs: Any
    ) -> AIMessage:
        """Invoke the mock LLM with input."""
        if isinstance(input, str):
            response = self._generate_mock_response(input)
        else:
            response = self._generate_mock_response(str(input))

        return AIMessage(content=response)

    async def ainvoke(
        self, input: Any, config: Optional[Any] = None, **kwargs: Any
    ) -> AIMessage:
        """Async invoke the mock LLM with input."""
        return self.invoke(input, config, **kwargs)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response based on the prompt content."""
        prompt_lower = prompt.lower()

        if "extract" in prompt_lower and "entities" in prompt_lower:
            return self._mock_entity_extraction()
        elif "summary" in prompt_lower or "analyze" in prompt_lower:
            return self._mock_document_summary()
        else:
            return self._mock_general_response()

    def _mock_entity_extraction(self) -> str:
        """Generate mock entity extraction response."""
        return """
Type: law
Value: Civil Rights Act
Confidence: 0.95
Source: Section 1 of the Civil Rights Act
Start: 150
End: 167
---
Type: agency
Value: Department of Justice
Confidence: 0.90
Source: enforcement shall be handled by the Department of Justice
Start: 342
End: 362
---
Type: affected_group
Value: Protected Classes
Confidence: 0.85
Source: individuals in protected classes
Start: 489
End: 510
---
Type: legal_concept
Value: Due Process
Confidence: 0.88
Source: due process requirements under the Fourteenth Amendment
Start: 654
End: 703
---
"""

    def _mock_document_summary(self) -> str:
        """Generate mock document summary response."""
        return """
# Executive Summary

This document establishes comprehensive civil rights protections and enforcement mechanisms. The legislation aims to prevent discrimination and ensure equal treatment under the law for all individuals regardless of protected characteristics.

The key provisions focus on enforcement procedures, compliance requirements, and remedial actions for violations. The document demonstrates strong constitutional foundation and aligns with existing civil rights jurisprudence.

# Key Points

• Establishes clear anti-discrimination standards
• Creates enforcement mechanisms through federal agencies
• Provides private right of action for violations
• Includes comprehensive compliance monitoring
• Addresses both individual and systemic discrimination

# Main Provisions

**Section 1 - Prohibited Conduct**
- Impact: High - Establishes fundamental protections
- Scope: Applies to employment, housing, education, and public accommodations

**Section 2 - Enforcement Authority**
- Impact: Medium - Defines agency roles and responsibilities
- Timeline: 90 days for investigation completion

**Section 3 - Remedies**
- Impact: High - Provides meaningful relief for violations
- Includes monetary damages and injunctive relief

# Risk Assessments

**Civil Rights Implications: LOW RISK**
- Enhances rather than restricts civil liberties
- Provides additional protections for vulnerable populations

**Privacy Concerns: MEDIUM RISK**
- Investigation procedures may require disclosure of personal information
- Recommend additional privacy safeguards

**Constitutional Considerations: LOW RISK**
- Well-grounded in Commerce Clause and Fourteenth Amendment
- Consistent with existing Supreme Court precedent

# Affected Groups

- Racial and ethnic minorities
- Women and gender minorities
- Individuals with disabilities
- Religious minorities
- LGBTQ+ individuals

# Legal Precedents

- Brown v. Board of Education (1954)
- Heart of Atlanta Motel v. United States (1964)
- McDonnell Douglas Corp. v. Green (1973)

# Implementation Timeline

- Effective Date: 180 days after enactment
- Agency rulemaking: 90 days
- Training programs: 120 days
- Full compliance required: 365 days

# Overall Confidence Score

0.92 - High confidence based on clear legal language, strong constitutional foundation, and comprehensive enforcement mechanisms.
"""

    def _mock_general_response(self) -> str:
        """Generate mock general response."""
        return """
This is a mock response for testing purposes. The document appears to be a legal or policy document that would typically require detailed analysis of its provisions, implications, and compliance requirements.

Key areas for analysis would include:
- Legal authority and constitutional basis
- Implementation requirements
- Enforcement mechanisms
- Affected parties and stakeholders
- Potential legal challenges
- Compliance timeline and procedures

For production use, please configure valid API keys for OpenAI or Anthropic to enable full analysis capabilities.
"""

    @property
    def _llm_type(self) -> str:
        """Return the LLM type."""
        return "mock"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters."""
        return {"model_name": self.model_name}

    def generate_prompt(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Generate responses for a list of prompts."""
        generations = []
        for prompt in prompts:
            response = self._generate_mock_response(prompt)
            generations.append([Generation(text=response)])
        return LLMResult(generations=generations)

    async def agenerate_prompt(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Async generate responses for a list of prompts."""
        return self.generate_prompt(prompts, stop)

    def predict(self, text: str, stop: Optional[List[str]] = None) -> str:
        """Predict response for a single text input."""
        return self._generate_mock_response(text)

    async def apredict(self, text: str, stop: Optional[List[str]] = None) -> str:
        """Async predict response for a single text input."""
        return self.predict(text, stop)

    def predict_messages(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> BaseMessage:
        """Predict response for a list of messages."""
        last_message = messages[-1].content if messages else ""
        response = self._generate_mock_response(last_message)
        return AIMessage(content=response)

    async def apredict_messages(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> BaseMessage:
        """Async predict response for a list of messages."""
        return self.predict_messages(messages, stop)
