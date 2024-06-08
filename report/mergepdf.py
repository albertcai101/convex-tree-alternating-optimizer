import os
from glob import glob
from pypdf import PdfMerger

allpdfs = ['writeup.pdf']
targetpdf = 'Submission v{}.pdf'

def find_files():
    found_files = []
    for pdf in allpdfs:
        search_pattern = f"**/{pdf}"
        matches = glob(search_pattern, recursive=True)
        if matches:
            found_files += matches
    return found_files


def merge_files(pdfpaths):
    merger = PdfMerger()

    [merger.append(pdf) for pdf in pdfpaths]

    find_counter()

    with open(targetpdf, "wb") as new_file:
        merger.write(new_file)

    merger.close()

def find_counter():
    counter = 0
    global targetpdf
    while os.path.isfile(targetpdf.format(counter)):
        counter += 1
    targetpdf = targetpdf.format(counter)

if __name__ == "__main__":
    print(find_files())
    merge_files(find_files())