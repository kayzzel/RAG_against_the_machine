from .models import Chunk

import ast
import re


class Chunking:
    """Splits source files into chunks for indexing.

    Attributes:
        max_chunk_size: Maximum chunk size in characters.
        overlap: Number of characters shared between consecutive chunks.
    """

    def __init__(self, max_chunk_size: int = 2000, overlap: int = 150):
        """Initialize the chunking configuration.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            overlap: Number of characters shared between consecutive chunks.
        """
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")
        if overlap >= max_chunk_size:
            raise ValueError("overlap must be smaller than max_chunk_size")

        self.__max_chunk_size = max_chunk_size
        self.__overlap = overlap
        self.__HEADER_RE = re.compile(r"^(#{1,6})\s+.*$")

    def basic_chunking(self, filename: str, source: str) -> list[Chunk]:
        """Split source text into fixed-size chunks with overlap.

        Args:
            filename: Path to the source file.
            source: Full text content of the file.

        Returns:
            List of Chunk objects with positions in the original file.
        """
        index: int = 0
        chunks: list[Chunk] = []
        unchuncked_size: int = len(source)

        while unchuncked_size > 0:
            if unchuncked_size <= self.__max_chunk_size:
                chunks.append(Chunk(
                    content=source[index:index+unchuncked_size],
                    file_path=filename,
                    start_index=index,
                    end_index=index+unchuncked_size,
                    chunk_type="basic"
                ))
                break

            chunks.append(Chunk(
                content=source[index:index+self.__max_chunk_size],
                file_path=filename,
                start_index=index,
                end_index=index+self.__max_chunk_size,
                chunk_type="basic"
            ))

            unchuncked_size -= self.__max_chunk_size - self.__overlap
            index += self.__max_chunk_size - self.__overlap

        return chunks

    def __append_chunk_or_split(
                    self,
                    chunks: list[Chunk],
                    filename: str,
                    source: str,
                    start: int,
                    end: int,
                    chunk_type: str
                ) -> None:
        """Append a chunk from source[start:end], splitting with
        basic_chunking if it exceeds max_chunk_size.

        Args:
            chunks: List to append the resulting chunk(s) to.
            filename: Path to the source file.
            source: Full text content of the file.
            start: Starting character index of the range.
            end: Ending character index of the range.
            chunk_type: Type to assign to the appended chunk(s).
        """
        text = source[start:end]
        if len(text) <= self.__max_chunk_size:
            chunks.append(Chunk(
                content=text,
                file_path=filename,
                start_index=start,
                end_index=end,
                chunk_type=chunk_type,
            ))
            return

        sub_chunks = self.basic_chunking(filename, text)
        for sub in sub_chunks:
            sub.start_index += start
            sub.end_index += start
            sub.chunk_type = chunk_type
            chunks.append(sub)

    def __find_sections(self, source: str) -> list[tuple[int, int, int]]:
        """Find header sections as (depth, start_char, end_char) tuples.

        depth=0 is used for any leading content before the first header.

        Args:
            source: Full Markdown text content.

        Returns:
            List of (depth, start_char, end_char) for each section.
        """
        lines = source.splitlines(keepends=True)
        sections: list[tuple[int, int]] = []  # (depth, start_char)
        offset = 0
        in_code_fence = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence

            match = self.__HEADER_RE.match(line) if not in_code_fence else None
            if match:
                depth = len(match.group(1))
                sections.append((depth, offset))

            offset += len(line)

        if not sections or sections[0][1] != 0:
            sections.insert(0, (0, 0))

        result = []
        for i, (depth, start) in enumerate(sections):
            end = sections[i + 1][1] if i + 1 < len(sections) else len(source)
            result.append((depth, start, end))
        return result

    def __merge_by_depth(
                    self,
                    sections: list[tuple[int, int, int]]
                 ) -> list[tuple[int, int]]:
        """Merge sections so each entry spans a header and everything under
        it until the next header of equal-or-shallower depth than the
        deepest section absorbed so far.

        Args:
            sections: List of (depth, start_char, end_char) tuples.

        Returns:
            List of (start_char, end_char) tuples for merged sections.
        """
        merged: list[tuple[int, int]] = []
        parent_index = 0

        while parent_index < len(sections):
            parent_depth, parent_start, _ = sections[parent_index]

            child_index = parent_index + 1
            while (
                        child_index < len(sections) and
                        sections[child_index][0] > parent_depth
                    ):
                parent_depth = sections[child_index][0]
                child_index += 1

            last_absorbed = sections[child_index - 1]
            parent_end = last_absorbed[2]

            merged.append((parent_start, parent_end))
            parent_index = child_index

        return merged

    def markdown_chunking(self, filename: str, source: str) -> list[Chunk]:
        """Chunk a Markdown file by header sections.

        Falls back to basic_chunking for any section exceeding
        max_chunk_size.

        Args:
            filename: Path to the source file.
            source: Full Markdown text content.

        Returns:
            List of Chunk objects with positions in the original file.
        """
        raw_sections = self.__find_sections(source)
        merged = self.__merge_by_depth(raw_sections)

        chunks: list[Chunk] = []
        for start, end in merged:
            self.__append_chunk_or_split(
                    chunks,
                    filename,
                    source,
                    start,
                    end,
                    "markdown"
                )
        return chunks

    def __build_line_offsets(self, source: str) -> list[int]:
        """Compute the character index where each line starts.

        Args:
            source: Full text content.

        Returns:
            List where offsets[i] is the character index at which line
            i+1 (1-indexed) starts.
        """
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def __node_char_range(
        self, node: ast.stmt, line_offsets: list[int]
    ) -> tuple[int, int]:
        """Compute the character (start, end) range for a top-level
        statement, extending the start to cover any decorators so they
        stay attached to their def.

        Args:
            node: Top-level statement node from the AST.
            line_offsets: Line start offsets from __build_line_offsets.

        Returns:
            (start_char, end_char) tuple for the statement.

        Raises:
            ValueError: If the node is missing end position information.
        """
        start_node: ast.expr | ast.stmt = node
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            start_node = decorators[0]

        if node.end_lineno is None or node.end_col_offset is None:
            raise ValueError(
                f"AST node {node!r} is missing end position information"
            )

        start = line_offsets[start_node.lineno - 1] + start_node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        return start, end

    def python_chunking(self, filename: str, source: str) -> list[Chunk]:
        """Chunk a Python file using AST boundaries.

        Each top-level function or class becomes its own chunk (falling
        back to basic_chunking if oversized); consecutive non-def/class
        statements are grouped together.

        Args:
            filename: Path to the source file.
            source: Full Python source text.

        Returns:
            List of Chunk objects with positions in the original file.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self.basic_chunking(filename, source)

        line_offsets = self.__build_line_offsets(source)
        chunks: list[Chunk] = []
        loose_run: list[ast.stmt] = []

        def flush_loose_run() -> None:
            if not loose_run:
                return
            start, _ = self.__node_char_range(loose_run[0], line_offsets)
            _, end = self.__node_char_range(loose_run[-1], line_offsets)
            self.__append_chunk_or_split(
                    chunks,
                    filename, source, start, end, "code"
                                         )
            loose_run.clear()

        for node in tree.body:
            if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                flush_loose_run()
                start, end = self.__node_char_range(node, line_offsets)
                self.__append_chunk_or_split(
                        chunks,
                        filename,
                        source,
                        start,
                        end,
                        "code"
                    )
            else:
                loose_run.append(node)

        flush_loose_run()
        return chunks

    def chunk_file(self, filename: str) -> list[Chunk]:
        """Chunk a file based on its type.

        Python files use AST-aware chunking, Markdown files use header
        sections, and everything else falls back to basic chunking.

        Args:
            filename: Path to the file to chunk.

        Returns:
            List of Chunk objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read.
        """
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()

        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix == "py":
            return self.python_chunking(filename, source)
        if suffix in ("md", "markdown"):
            return self.markdown_chunking(filename, source)
        return self.basic_chunking(filename, source)
