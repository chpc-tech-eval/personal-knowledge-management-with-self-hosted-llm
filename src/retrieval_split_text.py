"""
# Split documents into chunks

## This provides the `split_source_text_in_dir` function

For effective RAG, we want to given the model relevant information without too much irrelevant information (which wastes its context window and may mislead the model).

See [the document database update script](retrieval_db_update.ipynb) for more background.

## Supports Markdown

Currently the only document format supported is Markdown. This can be expanded as necessary.
"""

import os
import unicodedata
import re

def to_file_name(value: str):
    """
    Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase.
    Also strip leading and trailing whitespace, dashes, and underscores.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def split_source_text(path: str, softmax: int = 4096):
    filename = os.path.basename(path)
    filename_no_ext, file_extension = os.path.splitext(filename)

    filenames = []
    sections = []

    if file_extension == ".md":
        # split by headings, making sure to include the source information
        # for example, an output might look like
        """
        file: tutorial1_README.md
        # Network Primer
        ## Basic Networking Example (WhatIsMyIp.com)
        [part 1]

        In the following examples, you will be using your ...
        """

        file_head = f"file: {filename}"
        headers = ["", "", ""]
        part = 1

        section = ""

        incodeblock = False

        append_filename = lambda h, p : filenames.append(to_file_name(f"{filename_no_ext}-{h[0]}-{h[1]}-{h[2]}-{p}") + file_extension)
        append_section = lambda h, p, s : sections.append(f"{file_head}\n{h[0]}\n{h[1]}\n{h[2]} [part {p}]\n{s}")

        with open(path, 'r') as file:
            for line in file:
                stripped = line.strip()

                if not incodeblock:
                    if not stripped:
                        if len(section) >= softmax:
                            append_filename(headers, part)
                            append_section(headers, part, section)
                            section = ""
                            part += 1
                            
                    
                    numhashes = min(len(stripped) - len(stripped.lstrip('#')), 3)

                    if numhashes > 0 and numhashes <= 3:
                        if section:
                            append_filename(headers, part)
                            append_section(headers, part, section)
                            section = ""
                            part = 1
                    
                        headers[numhashes - 1] = stripped
                        for i in range(numhashes, 3):
                            headers[i] = ""
                    else:
                        section += line
                else:
                    section += line

                incodeblock = incodeblock ^ (stripped.count("```") % 2 == 1)
               
        if section:
            append_filename(headers, part)
            append_section(headers, part, section)

        return filenames, sections

        
    else:
        raise Exception(f"Unknown file type {file_extension}")


def split_source_text_in_dir(srcdir: str, dstdir: str):
    if not os.path.isdir(srcdir):
        raise Exception("Dir not found: f{srcdir}")

    if not os.path.isdir(dstdir):
        os.mkdir(dstdir)

    for entry in os.scandir(srcdir):  
        if entry.is_file():
            filenames, sections = split_source_text(entry.path)

            for i in range(len(sections)):
                 with open(os.path.join(dstdir, filenames[i]), "w") as f:
                    f.write(sections[i])

