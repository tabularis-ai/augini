#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Installing mkdocs-material...${NC}"
pip install mkdocs-material

echo -e "${GREEN}Starting local documentation server...${NC}"
echo -e "${YELLOW}Visit http://127.0.0.1:8000 to view the documentation${NC}"
mkdocs serve 