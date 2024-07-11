import os
import argparse

def process_directory(directory_path, output_file, extensions):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Check if file has one of the specified extensions
                if extensions and not any(file.endswith(ext) for ext in extensions):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory_path)
                
                # Write the file path as a Markdown header
                outfile.write(f"## {relative_path}\n\n")
                
                # Write the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write("```\n")
                        outfile.write(content)
                        outfile.write("\n```\n\n")
                except UnicodeDecodeError:
                    outfile.write(f"Error reading file: {file_path} (Unicode Decode Error)\n\n")
                except Exception as e:
                    outfile.write(f"Error reading file: {file_path} ({str(e)})\n\n")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process a directory and output its contents to a Markdown file.')
    parser.add_argument('-d', '--directory', type=str, help='Directory to process', default=os.getcwd())
    parser.add_argument('-o', '--output', type=str, help='Output Markdown file', default='directory_contents.md')
    parser.add_argument('-e', '--extensions', type=str, nargs='*', help='File extensions to include (e.g., py js)')

    # Parse arguments
    args = parser.parse_args()
    directory_to_process = args.directory
    output_markdown_file = args.output
    file_extensions = args.extensions

    # Debugging: Print starting process
    print(f"Processing directory: {directory_to_process}")
    print(f"File extensions: {file_extensions}")

    if not os.path.isdir(directory_to_process):
        print(f"Error: The directory '{directory_to_process}' does not exist.")
    else:
        process_directory(directory_to_process, output_markdown_file, file_extensions)
        print(f"Markdown file '{output_markdown_file}' has been created with the contents of '{directory_to_process}'.")
