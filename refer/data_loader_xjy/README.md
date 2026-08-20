This module is used as a tool to read binary files of different kind.

## Usage

Import DataPool from data_pool to initialize data loader.

DataPool requires 3 parameter.

data_path: Path object that record the position the data is stored.

name_dict: A dictionary that contains the asset, sheet and field user want to get.

sheet_loader_dict: A dictionary that stores the sheet to loader relationship. Stored in data/sheet_config.json

## Test
Run with uv run demo/demo.py.

The test collect the close data of daily stock and uses three type of reading operation.

1. single day reading using memmap
2. multiple day reading using memmap
3. read array and return a pandas dataframe
