#!/bin/bash
# Script to publish the Medical ASR Evaluator repository to GitHub

set -e

echo "🚀 Publishing Medical ASR Evaluator to GitHub"
echo "=============================================="
echo ""

# Check if remote already exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "✓ Remote 'origin' already configured:"
    git remote get-url origin
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter GitHub repository URL (e.g., git@github.com:username/Medical_ASR_Evaluator.git): " REPO_URL
        git remote set-url origin "$REPO_URL"
    fi
else
    read -p "Enter GitHub repository URL (e.g., git@github.com:username/Medical_ASR_Evaluator.git): " REPO_URL
    git remote add origin "$REPO_URL"
fi

echo ""
echo "📤 Pushing to GitHub..."
echo ""

# Push to GitHub
git push -u origin main

echo ""
echo "✅ Successfully published to GitHub!"
echo ""
echo "Next steps:"
echo "  1. Visit your repository on GitHub"
echo "  2. Add a description and topics"
echo "  3. Consider adding GitHub Actions for CI/CD"
echo "  4. Update README with your repository URL"

