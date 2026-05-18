from datasets import load_dataset


class DatasetAnalyzer:
    """Loads and provides analysis utilities for the Bitext customer support dataset."""

    DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

    def __init__(self):
        ds = load_dataset(self.DATASET_NAME)
        self.data = ds["train"].to_pandas()

    def get_categories(self) -> list[str]:
        """Return a sorted list of all unique category values in the dataset.

        Returns:
            List of category strings, e.g. ['ACCOUNT', 'DELIVERY', 'FEEDBACK', ...].
        """
        return sorted(self.data["category"].unique().tolist())

    def get_intents(self, category: str | None = None) -> list[str]:
        """Return a sorted list of unique intents, optionally scoped to a category.

        Args:
            category: If provided, only return intents belonging to this category.
                      Must match the exact string stored in the dataset (case-sensitive).

        Returns:
            List of intent strings, e.g. ['cancel_order', 'track_refund', ...].

        Raises:
            ValueError: If the given category does not exist in the dataset.
        """
        df = self.data
        if category is not None:
            if category not in self.data["category"].values:
                raise ValueError(f"Category '{category}' not found. Available: {self.get_categories()}")
            df = df[df["category"] == category]
        return sorted(df["intent"].unique().tolist())

    def get_distribution(self, column: str) -> dict[str, int]:
        """Return the value counts for a given column as a dict sorted by frequency (descending).

        Args:
            column: Name of the column to compute the distribution for.
                    Common values: 'category', 'intent', 'flags'.

        Returns:
            Dict mapping each unique value to its count, e.g. {'ACCOUNT': 1200, 'SHIPPING': 980, ...}.

        Raises:
            ValueError: If the column does not exist in the dataset.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found. Available columns: {list(self.data.columns)}")
        counts = self.data[column].value_counts()
        return counts.to_dict()

    def get_examples(self, n: int = 5, **filters) -> list[dict]:
        """Return up to n rows from the dataset, optionally filtered by column values.

        Args:
            n: Maximum number of examples to return.
            **filters: Keyword arguments used to filter rows by exact column match.
                       e.g. category="SHIPPING", intent="track_order".

        Returns:
            List of dicts, each representing one row of the dataset.

        Raises:
            ValueError: If a filter key is not a valid column name.
        """
        df = self.data
        for col, val in filters.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
            df = df[df[col] == val]
        return df.head(n).to_dict(orient="records")

    def count(self, **filters) -> int:
        """Count rows in the dataset, optionally filtered by column values.

        Args:
            **filters: Keyword arguments used to filter rows by exact column match.
                       e.g. category="ACCOUNT", intent="cancel_order".
                       If no filters are given, returns the total number of rows.

        Returns:
            Integer count of matching rows.

        Raises:
            ValueError: If a filter key is not a valid column name.
        """
        df = self.data
        for col, val in filters.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
            df = df[df[col] == val]
        return len(df)

    def search(self, keyword: str, column: str = "instruction") -> list[dict]:
        """Search for rows where a column contains a given keyword (case-insensitive).

        Args:
            keyword: The substring to search for.
            column: The column to search in. Defaults to 'instruction' (customer message).

        Returns:
            List of dicts for all matching rows.

        Raises:
            ValueError: If the column does not exist in the dataset.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found. Available columns: {list(self.data.columns)}")
        mask = self.data[column].str.contains(keyword, case=False, na=False)
        return self.data[mask].to_dict(orient="records")

    def get_stats(self) -> dict:
        """Return a high-level summary of the dataset.

        Returns:
            Dict with the following keys:
            - 'num_rows': total number of rows
            - 'columns': list of column names
            - 'unique_counts': dict mapping each column to the number of unique values
            - 'avg_instruction_length': average character length of customer messages
            - 'avg_response_length': average character length of agent responses
        """
        stats = {
            "num_rows": len(self.data),
            "columns": list(self.data.columns),
            "unique_counts": {col: self.data[col].nunique() for col in self.data.columns},
        }
        if "instruction" in self.data.columns:
            stats["avg_instruction_length"] = self.data["instruction"].str.len().mean()
        if "response" in self.data.columns:
            stats["avg_response_length"] = self.data["response"].str.len().mean()
        return stats
