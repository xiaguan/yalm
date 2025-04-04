#!/bin/bash

# Script to format all C/C++ files using clang-format

# Find all C/C++ files
FILES=$(find ./src -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.c" -o -name "*.hpp" \))

# not format json.hpp
FILES=$(echo "$FILES" | grep -v "json.hpp")

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format is not installed. Please install it first."
    exit 1
fi

# Format each file
echo "Formatting files..."
for FILE in $FILES; do
    echo "Formatting $FILE"
    clang-format -i -style=file "$FILE"
done

echo "All files have been formatted according to .clang-format configuration."
