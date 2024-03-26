import os
import fitz
import pandas as pd
import numpy as np
from PIL import Image
from skimage.measure import find_contours
import re
import string

def detect_black_boxes(page):
    """
    Detect black boxes within the PDF page.
    """
    black_boxes = []
    # Convert the page to an image and check for black regions
    image = page.get_pixmap()
    black_regions = find_black_regions(image)
    # Extract text from black regions
    for bbox in black_regions:
        text = page.get_text("text", clip=bbox)
        if text.strip().startswith('P') and len(text.strip()) > 20 and not text.strip().startswith('POLE'):
            matches = re.findall(r'\bP\d+\b', text)
            if matches:
                lines = text.strip().split('\n')
                filtered_lines = [line for line in lines if not any(word in line for word in ['DETAILS', 'NUMBER', 'NAME'])]
                filtered_text = '\n'.join(filtered_lines)
                black_boxes.append((filtered_text, bbox))
    return black_boxes

def find_black_regions(image):
    """
    Find black regions within an image.
    """
    # Convert the image to grayscale using Pillow
    pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
    gray_image = pil_image.convert("L")
    # Threshold the image to obtain binary image
    binary_image = np.array(gray_image.point(lambda x: 0 if x < 200 else 255, '1'))
    # Get contours
    contours = find_contours(binary_image, 0.5)
    # Get bounding boxes of contours
    bounding_boxes = []
    for contour in contours:
        min_row, min_col = contour.min(axis=0)
        max_row, max_col = contour.max(axis=0)
        bounding_boxes.append((min_col, min_row, max_col, max_row))
    return bounding_boxes

def extract_dwg_number(page):
    """
    Extract the DWG number located at the bottom right of the PDF page.
    """
    dwg_number = '2021-019060'
    dwg_number_pattern = re.compile(r'dwg\.?\s*no\.?\s*(\d+-\d+)', re.IGNORECASE)
    text = page.get_text()
    match = dwg_number_pattern.search(text)
    if match:
        dwg_number = match.group(1)
    else:
        print(f"DWG number not found in page {page.number + 1}.")
    return dwg_number


def save_to_text_file(data, dwg_number):
    """
    Save extracted data to a text file based on DWG number.
    """
    output_file = f'{dwg_number}.txt'
    seen_boxes = set()

    with open(output_file, 'w', encoding='utf-8') as file:
        for i, (box_data, _) in enumerate(data, start=1):
            # Remove illegal characters
            box_data = ''.join(filter(lambda x: x in string.printable, box_data))

            # Extract the box name
            box_name = re.search(r'(P\d+)', box_data)
            if box_name:
                box_name = box_name.group(1)
                if box_name not in seen_boxes:
                    seen_boxes.add(box_name)
                    # Correct formatting for box name
                    box_data = box_data.replace(box_name, f"Box {box_name}")

                    # General corrections (applied to all boxes)
                    box_data = re.sub(r"45'\s+W\(30'\s+WJ", "45'W(30'W)", box_data)
                    box_data = re.sub(r"45'\s+W\(O'\s*WJ?\)", "45'W(40'W)", box_data)
                    box_data = box_data.replace("WJ", "W)")
                    box_data = re.sub(r"45'W\(O'\s*\n?\s*W\)", "45'W(O'W)", box_data)
                    box_data = re.sub(r"45'\s*\nW\(40'W\)", "45'W(40'W)", box_data)
                    box_data = re.sub(r"45'\s*\n?\s*W\(40'\s*\n?\s*W\)", "45'W(40'W)", box_data)
                    box_data = re.sub(f"{box_name}\(Box {box_name}\)", f"{box_name}({box_name})", box_data)
                    box_data = box_data.replace(" & \n", "&")
                    box_data = box_data.replace("?", "7")
                    box_data = re.sub(r'[_\-<o]+$', '', box_data, flags=re.MULTILINE)
                    box_data = re.sub(r'r\-+I$', '', box_data, flags=re.MULTILINE)
                    box_data = box_data.replace("B 1201240V", "B 120/240V")
                    box_data = box_data.replace("30.;JSOB", "30-36OB")
                    box_data = re.sub(r'\s*\d+\s*$', '', box_data)
                    box_data = re.sub(r"45'\s*W\(35\"?W\)", "45'W(35'W)", box_data)
                    box_data = re.sub(r"45'\s*\n?\s*W\((\d+'?)W\)", r"45'W(\1W)", box_data)
                    box_data = re.sub(r'(3&)\s+(\d)', r'\1\2', box_data)
                    box_data = re.sub(r'\n[_\-]+', '\n', box_data)
                    box_data = re.sub(r'(SEE NOTE\s*\d+)(\s*&\s*\n\s*)(\d+)', r'\1 & \3', box_data, flags=re.DOTALL)
                    box_data = re.sub(r'[_\,]+\n', '\n', box_data)
                    box_data = re.sub(r'Box P201 :\s*\n\s*\(\s*Box P201\s*\)', 'Box P201(P201):', box_data)
                    box_data = re.sub(r'\n\s*,', '\n', box_data)
                    box_data = re.sub(r'\n\s+', '\n', box_data)  
                    box_data = re.sub(r'(SEE NOTE 1, 2),\s*\n(\d+)\s*\n&(\d+)', r'\1, \2 &\3', box_data)
                    box_data = re.sub(r'Box P201 :\s*\n\s*\(\s*Box P201\s*\)', 'Box P201(P201)', box_data)
                    box_data = box_data.replace('(Box P201)', '')  
                    box_data = re.sub(r'(SEE NOTE 1, 2),\s*(3 &4)', r'\1,\n\2', box_data)  
                    box_data = re.sub(r'Box P201 :\s*\n\s*\(\s*Box P201\s*\)\s*', 'Box P201(P201)', box_data)
                    box_data = re.sub(r',i\s*$', '', box_data)
                    box_data = re.sub(r"(O')\s*\n\s*(W\))", r"\1\2", box_data)
                    
                    if box_name == 'P125':
                        box_data = box_data.replace('05-1', '05-100D') 
                    if box_name == 'P141':
                        box_data = box_data.replace('1-100(', '1-100(1-50)')  
                    if box_name == 'P195':
                        box_data = box_data.replace("\n \n,05-100D", " 05-100D")
                    if box_name == 'P201':
                        box_data = re.sub(r'^.*' + box_name + '.*:', f"Box {box_name}({box_name})", box_data, flags=re.MULTILINE)   
                        box_data = re.sub(r'Box\s+' + box_name, f"Box {box_name}({box_name})", box_data, 1)
                        box_data = re.sub(rf'Box\s+{box_name}\s*:\s*', f'Box {box_name}({box_name})', box_data) 
                        box_data = re.sub(f"Box {box_name}\({box_name}\)\s*\n\s*\n", f"Box {box_name}({box_name})\n", box_data, 1)
                    if box_name == 'P222':
                        box_data = box_data.replace('I 11-413C', '11-413C')
                        box_data = box_data.replace('22 04-520P', '04-520P') 
                    if box_name == 'P218':
                        box_data = box_data.replace('l 09-950A', '09-950A')
                        box_data = box_data.replace('1 04-520P', '04-520P')
                        box_data = box_data.replace('L (SEE NOTE 1, 2', '(SEE NOTE 1, 2,')
                        box_data = box_data.replace('3, 4', '3, 4 & 5)')
                    if box_name == 'P212':
                        box_data = box_data.replace('4 05-100D', '05-100D')
                    # Format and filter data
                    box_data = box_data.strip().replace('\n', '\n\n', 1).replace('\n', ':\n', 1)
                    filtered_data = [line for line in box_data.split('\n') if line.strip() not in ['9', 'L', '97', '-', '5', '8', 'l', '.', '129', '12', 'ODD', '1-50)', 'I','1','_','195','n','=','220','PLACE','IN PLACE','&','5)'] and "107 " not in line]

                    # Join the filtered data back with newlines and write to file
                    filtered_box_data = '\n'.join(filtered_data)
                    file.write(f"{filtered_box_data.strip()}\n\n")

    print(f"Extracted data saved to '{output_file}'.")


def save_to_excel(data, excel_file):
    """
    Save extracted data to an Excel file.
    """
    df = pd.DataFrame(data, columns=['Content', 'Bounding Box'])
    # Remove illegal characters
    df['Content'] = df['Content'].apply(lambda x: ''.join(filter(lambda y: y in string.printable, x)))
    df['Box Number'] = df.index + 1
    # Reorder columns
    df = df[['Box Number', 'Content', 'Bounding Box']]
    df.to_excel(excel_file, index=False)
    print(f"Combined data saved to '{excel_file}'.")

def main():
    # Replace 'pdf_directory' with the directory containing the PDF files
    pdf_directory = 'C:/Users/User/Desktop/Python Assignment'
    combined_data = []
    for pdf_file in os.listdir(pdf_directory):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        if pdf_file.endswith('.pdf'):
            with fitz.open(pdf_path) as pdf_file:
                for page_num in range(len(pdf_file)):
                    page = pdf_file.load_page(page_num)
                    # Detect black boxes within the PDF page
                    black_boxes = detect_black_boxes(page)
                    if black_boxes:
                        dwg_number = extract_dwg_number(page)
                        save_to_text_file(black_boxes, dwg_number)
                        combined_data.extend(black_boxes)
                        print(f"Extracted data from page {page_num + 1} in '{pdf_file}'.")

    # Save combined data to an Excel file
    excel_file = 'combined_data.xlsx'
    save_to_excel(combined_data, excel_file)

if __name__ == "__main__":
    main()
