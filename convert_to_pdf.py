"""
Simple markdown to PDF converter using markdown2pdf
"""
from pathlib import Path
import subprocess

def convert_md_to_pdf_simple():
    """Convert markdown to PDF using markdown-pdf"""
    
    md_file = 'research_paper.md'
    pdf_file = 'research_paper.pdf'
    
    print(f"Converting {md_file} to PDF...")
    
    try:
        # Try using markdown-pdf
        result = subprocess.run(
            ['markdown-pdf', md_file, '-o', pdf_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ PDF created successfully: {pdf_file}")
        else:
            print(f"Error: {result.stderr}")
            print("\nTrying alternative method...")
            convert_via_html()
            
    except FileNotFoundError:
        print("markdown-pdf not found, trying alternative method...")
        convert_via_html()

def convert_via_html():
    """Convert via HTML using markdown library"""
    import markdown
    
    # Read markdown
    with open('research_paper.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code']
    )
    
    # Create styled HTML
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Research Paper</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1 {{ font-size: 24px; margin-top: 30px; }}
            h2 {{ font-size: 20px; margin-top: 25px; }}
            h3 {{ font-size: 16px; margin-top: 20px; }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{ background-color: #f2f2f2; }}
            code {{
                background-color: #f5f5f5;
                padding: 2px 5px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f5f5f5;
                padding: 15px;
                border-left: 3px solid #ccc;
                overflow-x: auto;
            }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Save HTML
    with open('research_paper.html', 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print("✓ HTML version created: research_paper.html")
    print("\nTo convert to PDF:")
    print("1. Open research_paper.html in your browser")
    print("2. Press Ctrl+P (Print)")
    print("3. Select 'Save as PDF'")
    print("4. Save as research_paper.pdf")

if __name__ == "__main__":
    convert_md_to_pdf_simple()
