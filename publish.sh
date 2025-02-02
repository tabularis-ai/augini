#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting augini package publishing process...${NC}"

# Check if build and dist directories exist and remove them
echo -e "${YELLOW}Cleaning up old build artifacts...${NC}"
rm -rf build/ dist/ *.egg-info/

# Install/upgrade build tools
echo -e "${YELLOW}Upgrading build tools...${NC}"
python -m pip install --upgrade pip build twine

# Build the package
echo -e "${YELLOW}Building package...${NC}"
python -m build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed! Exiting...${NC}"
    exit 1
fi

# Verify the distribution files
echo -e "${YELLOW}Checking distribution with twine...${NC}"
python -m twine check dist/*

# Check if twine check was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Distribution files verified successfully!${NC}"
else
    echo -e "${RED}Distribution verification failed! Exiting...${NC}"
    exit 1
fi

# Ask for confirmation before uploading
echo -e "${YELLOW}Ready to upload to PyPI.${NC}"
read -p "Do you want to proceed with the upload? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Upload to PyPI
    echo -e "${YELLOW}Uploading to PyPI...${NC}"
    python -m twine upload dist/*
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Package successfully uploaded to PyPI!${NC}"
        
        # Get version from pyproject.toml
        VERSION=$(grep "version = " pyproject.toml | cut -d'"' -f2)
        echo -e "${GREEN}Version ${VERSION} has been published!${NC}"
        echo -e "${YELLOW}You can install it with: pip install augini==${VERSION}${NC}"
    else
        echo -e "${RED}Upload failed!${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Upload cancelled.${NC}"
    exit 0
fi 