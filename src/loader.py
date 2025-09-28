from pathlib import Path
#lang comm used
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader)



def load_document(file_paths):
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths] 

    docs = []
    for path in file_paths:
        path = str(path)
        ext = Path(path).suffix.lower()

        try:
            if ext == '.csv':
                loader = CSVLoader(path)
            elif ext in ['.xls', '.xlsx']:
                loader = UnstructuredExcelLoader(path)
            elif ext == '.txt':
                loader = TextLoader(path)        
            else:
                print(f"[WARN] Unsupported file type skipped: {path}")
                continue
            docs.extend(loader.load())

        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue

    return docs     