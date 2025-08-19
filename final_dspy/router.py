class Router:
    def __init__(self):
        self.llm_ready = False
        try:
            import dspy
            import os
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            if anthropic_key or openai_key:
                self.llm_ready = True
        except Exception as e:
            print(f"LLM config failed: {e}")
            self.llm_ready = False

    def classify(self, text: str) -> str:
        """
        Use LLM to classify the question as 'ANALYTIC', 'EXPLORATORY', or 'MIXED'.
        Fallback to rule-based if LLM is unavailable.
        """
        if self.llm_ready:
            import dspy
            prompt = (
                "You are a question classifier for a data analytics agent. "
                "Classify the following user question as one of: 'ANALYTIC', 'EXPLORATORY', or 'MIXED'. "
                "Return only the label, nothing else.\n\n"
                f"User question: {text}"
            )
            result = dspy.settings.lm(prompt)
            # Extract the label from the LLM response
            label = None
            if isinstance(result, str):
                label = result.strip().upper()
            elif isinstance(result, list) and result:
                label = str(result[0]).strip().upper()
            if label:
                if "ANALYTIC" in label:
                    return "ANALYTIC"
                if "EXPLORATORY" in label:
                    return "EXPLORATORY"
                if "MIXED" in label:
                    return "MIXED"
            # Fallback if LLM output is unexpected
            print(f"LLM classify fallback, got: {result}")
        # Rule-based fallback
        analytic_keywords = ["sum", "average", "top", "bottom", "filter", "show", "list", "count", "total", "max", "min", "order by", "limit"]
        exploratory_keywords = ["explain", "describe", "what is", "which column", "meaning", "card", "about"]
        text_lower = text.lower()
        if any(word in text_lower for word in analytic_keywords):
            if any(word in text_lower for word in exploratory_keywords):
                return "MIXED"
            return "ANALYTIC"
        if any(word in text_lower for word in exploratory_keywords):
            return "EXPLORATORY"
        return "ANALYTIC"  # Default
