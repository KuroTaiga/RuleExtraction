import re
from collections import Counter

def analyze_log_file(file_path):
    # Define the pattern to match "Z200_A*.mp4 的有效识别帧太少，无法生成可靠的分析结果"
    pattern = re.compile(r'Z200_A(\w+).mp4 的有效识别帧太少，无法生成可靠的分析结果')

    # Initialize a counter to store the counts of A* values
    a_star_counter = Counter()

    # Read the log file and process each line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Search for the pattern in the line
            match = pattern.search(line)
            if match:
                # Extract the A* part from the match (group(1) is the value after 'A')
                a_star_value = match.group(1)
                a_star_counter[a_star_value] += 1

    # Total count of lines with the matching pattern
    total_count = sum(a_star_counter.values())

    # Print the results
    print(f"Total count of lines with the matching pattern: {total_count}")
    print("Count of each A* value:")
    for a_star, count in a_star_counter.items():
        print(f"A{a_star}: {count}")

# Example usage
log_file_path = 'output.log'  # Replace this with the path to your log file
analyze_log_file(log_file_path)
