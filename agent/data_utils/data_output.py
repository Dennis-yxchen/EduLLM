import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

###### new change to pdf
# new function to generate pdf from json

def generate_pdf_from_json(json_file_path, pdf_file_path, title=None, footnotes=None):
    # add data from json file
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    # create a pdf file
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    
    def add_title(canvas, title_text):
        if title_text:
            canvas.saveState()
            canvas.setFont("Helvetica-Bold", 14)
            canvas.drawString(40, letter[1] - 30, title_text)
            canvas.line(40, letter[1] - 35, letter[0] - 40, letter[1] - 35)
            canvas.restoreState()
    
    def add_footer(canvas, footer_text, page_num):
        if footer_text:
            canvas.saveState()
            canvas.setFont("Helvetica", 9)
            canvas.drawString(40, 30, footer_text)
            canvas.line(40, 40, letter[0] - 40, 40)  # Move the line up
            canvas.drawString(letter[0] - 60, 30, f"Page {page_num}")
            canvas.restoreState()

    # write into data
    text = c.beginText(40, 710 if title else 750)  # Adjust starting position if there's a title
    text.setFont("Helvetica", 12)
    max_width = 500
    line_height = 14
    page_height = letter[1]
    # 设置底部边界，留出足够空间给脚注
    bottom_margin = 50 if footnotes else 40
    page_num = 1

    # Add title to first page only
    if title:
        add_title(c, title)

    for key, value in data.items():
        text.textLine(f"Question {int(key) + 1}:")
        text.setTextOrigin(40, text.getY() - line_height)
        lines = value.split('\n')
        for line in lines:
            if c.stringWidth(line, "Helvetica", 12) > max_width:
                words = line.split()
                current_line = ""
                for word in words:
                    if c.stringWidth(current_line + word + " ", "Helvetica", 12) <= max_width:
                        current_line += word + " "
                    else:
                        text.textLine(current_line)
                        current_line = word + " "
                        text.setTextOrigin(40, text.getY() - line_height)
                        if text.getY() < bottom_margin:  # 使用新的底部边界
                            c.drawText(text)
                            add_footer(c, footnotes, page_num)
                            c.showPage()
                            page_num += 1
                            text = c.beginText(40, 750)  # New pages start at the top
                            text.setFont("Helvetica", 12)
                text.textLine(current_line)
            else:
                text.textLine(line)
                text.setTextOrigin(40, text.getY() - line_height)
                if text.getY() < bottom_margin:  # 使用新的底部边界
                    c.drawText(text)
                    add_footer(c, footnotes, page_num)
                    c.showPage()
                    page_num += 1
                    text = c.beginText(40, 750)  # New pages start at the top
                    text.setFont("Helvetica", 12)
        text.setTextOrigin(40, text.getY() - line_height * 2)

    c.drawText(text)
    add_footer(c, footnotes, page_num)
    c.save()

    print(f"PDF file kept in {pdf_file_path}")