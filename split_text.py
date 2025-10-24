#!/usr/bin/env python3.11

import os
import unicodedata
import re

PRJ_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(PRJ_DIR, "data", "raw")
DST_DIR = os.path.join(PRJ_DIR, "data", "split")

def split_source_text(path: str):
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

        In the following examples, you will be using your ...
        """

        file_head = f"file: {filename}"
        h = ["", "", ""]

        section = ""

        incodeblock = False

        with open(path, 'r') as file:
            for line in file:
                stripped = line.strip()

                if not incodeblock:
                    numhashes = min(len(stripped) - len(stripped.lstrip('#')), 3)

                    if numhashes > 0 and numhashes <= 3:
                        if section:
                            filenames.append(to_file_name(f"{filename_no_ext}-{h[0]}-{h[1]}-{h[2]})") + file_extension)
                            sections.append(f"{file_head}\n{h[0]}\n{h[1]}\n{h[2]}\n{section}")
                            section = ""
                    
                        h[numhashes - 1] = stripped
                        for i in range(numhashes, 3):
                            h[i] = ""
                    else:
                        section += line
                else:
                    section += line

                incodeblock = incodeblock ^ (stripped.count("```") % 2 == 1)
               
        if section:
            filenames.append(to_file_name(f"{filename}_{h[0]}_{h[1]}_{h[2]})"))
            sections.append(f"{file_head}\n{h[0]}\n{h[1]}\n{h[2]}\n{section}")
            section = ""

        return filenames, sections

        
    else:
        raise Exception(f"Unknown file type {file_extension}")
    
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

def split_many(srcdir: str, dstdir: str):
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

def main():
    split_many(SRC_DIR, DST_DIR)

if __name__ == "__main__":
    main()
