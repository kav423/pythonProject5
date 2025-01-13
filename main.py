import re
import PyPDF2
import docx
import os
import logging
import markdown
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from xhtml2pdf import pisa
import io
import tempfile
from pdf2image import convert_from_path
import torch
import torchvision.transforms as transforms
import numpy as np
from transformers import SwinModel

# Загружаем предобученную модель Swin
try:
    swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
    swin_model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

except Exception as e:
    logging.error("Ошибка при загрузке модели Swin: " + str(e))
    print("Установите transformers и убедитесь, что модель Swin доступна.")


def get_image_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = swin_model(img_tensor).last_hidden_state
        return embedding.cpu().numpy()
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга изображения: {e}")
        return None


def markdown_to_image(markdown_text, output_path):
    try:
        html = markdown.markdown(markdown_text)
        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=pdf_buffer)

        if not pisa_status.err:
            # Создаем уникальную временную папку
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
                try:
                   with open(temp_pdf_path, 'wb') as temp_pdf:
                       temp_pdf.write(pdf_buffer.getvalue())

                   images = convert_from_path(temp_pdf_path, dpi=300,poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
                   if images:
                        images[0].save(output_path, 'PNG')
                        logging.info(f"Изображение MD успешно создано в '{output_path}'")

                        embedding = get_image_embedding(output_path)
                        if embedding is not None:
                            embedding_path = os.path.splitext(output_path)[0] + ".npy"
                            np.save(embedding_path, embedding)
                            logging.info(f"Эмбеддинг изображения сохранен в '{embedding_path}'")
                   else:
                      logging.error(f"Не удалось преобразовать PDF в изображение.")

                except Exception as e:
                    logging.error(f"Ошибка при конвертации: {e}")



        else:
            logging.error(f"Ошибка при конвертации MD в PDF: {pisa_status.err}")

    except Exception as e:
        logging.error(f"Ошибка при конвертации MD в изображение: {e}")


def analyze_and_convert_text(text):
    markdown_lines = []
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append("")
                continue

            # Обработка code blocks (с отступами или без)
            code_block_match = re.match(r"^( {0,3})```(.*)\n(.*?)\n( {0,3})```$", line, re.DOTALL)
            if code_block_match:
                indent1 = code_block_match.group(1) if code_block_match.group(1) else ""
                language = code_block_match.group(2)
                content = code_block_match.group(3)
                indent2 = code_block_match.group(4) if code_block_match.group(4) else ""
                markdown_lines.append(f"{indent1}```{language}\n{content}\n{indent2}```")
                continue

            # Обработка заголовков (1-6 уровней)
            header_match = re.match(r'^(#+) (.*)$', line)
            if header_match:
                level = len(header_match.group(1))
                content = header_match.group(2)
                if 1 <= level <= 6:
                    markdown_lines.append(f'{"#" * level} {content.strip()}')
                    continue

            # Обработка нумерованных списков
            numbered_list_match = re.match(r'^(\d+)\.\s+(.*)$', line)
            if numbered_list_match:
                number = numbered_list_match.group(1)
                content = numbered_list_match.group(2)
                markdown_lines.append(f'{number}. {content}')
                continue

            # Обработка ненумерованных списков
            unordered_list_match = re.match(r'^([*+-])\s+(.*)$', line)
            if unordered_list_match:
                marker = unordered_list_match.group(1)
                content = unordered_list_match.group(2)
                markdown_lines.append(f'{marker} {content}')
                continue

            # Обработка цитат
            quote_match = re.match(r'^>\s+(.*)$', line)
            if quote_match:
                content = quote_match.group(1)
                markdown_lines.append(f'> {content.strip()}')
                continue

            # Обработка ссылок
            link_match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', line)
            if link_match:
                text = link_match.group(1)
                url = link_match.group(2)
                markdown_lines.append(f'[{text}]({url})')
                continue

            # Oбработка жирного и курсивного шрифта
            line = re.sub(r'\*\*(.*?)\*\*', r'**\1**', line)  # Жирный шрифт
            line = re.sub(r'\*(.*?)\*', r'*\1*', line)  # Курсив
            line = re.sub(r'_(.*?)_', r'_\1_', line)  # Курсив с подчеркиванием

            # Обработка горизонтальных линий
            hr_match = re.match(r'^[-*_]{3,}$', line)
            if hr_match:
                markdown_lines.append("---")
                continue

            # Обработка таблиц
            if re.match(r'^\.\.\.(.*?)\.\.\.$', line):
                markdown_lines.append(line)
                continue

            # Обычный текст с отступами
            indent = len(line) - len(line.lstrip())
            markdown_lines.append(" " * indent + line.strip())
        markdown_lines.append("")

    return '\n'.join(markdown_lines).strip()


def pdf_to_markdown(pdf_path, output_dir):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                markdown_text = analyze_and_convert_text(text)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.md")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text)
                logging.info(f"Страница {page_num + 1} из PDF '{pdf_path}' сконвертирована в '{output_path}'")
                # Создание PNG
                image_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.png")
                markdown_to_image(markdown_text, image_path)
                logging.info(f"Изображение страницы {page_num + 1} из PDF '{pdf_path}' создано в '{image_path}'")
    except Exception as e:
        logging.error(f"Ошибка при конвертации PDF: {e}")


def docx_to_markdown(docx_path, output_dir):
    try:
        doc = docx.Document(docx_path)
        for page_num, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text
            markdown_text = analyze_and_convert_text(text)
            base_name = os.path.splitext(os.path.basename(docx_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            logging.info(f"Параграф {page_num + 1} из DOCX '{docx_path}' сконвертирован в '{output_path}'")
            # Создание PNG
            image_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.png")
            markdown_to_image(markdown_text, image_path)
            logging.info(f"Изображение параграфа {page_num + 1} из DOCX '{docx_path}' создано в '{image_path}'")
    except Exception as e:
        logging.error(f"Ошибка при конвертации DOCX: {e}")


def txt_to_markdown(txt_path, output_dir):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            lines = text.split("\n")
            for page_num, line in enumerate(lines):
                text = line
                markdown_text = analyze_and_convert_text(text)
                base_name = os.path.splitext(os.path.basename(txt_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.md")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text)
                logging.info(f"Строка {page_num + 1} из TXT '{txt_path}' сконвертирована в '{output_path}'")
                # Создание PNG
                image_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.png")
                markdown_to_image(markdown_text, image_path)
                logging.info(f"Изображение строки {page_num + 1} из TXT '{txt_path}' создано в '{image_path}'")
    except Exception as e:
        logging.error(f"Ошибка при конвертации TXT: {e}")


def select_file():
    file_path = filedialog.askopenfilename(title="Выберите файл")
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)


def select_directory():
    dir_path = filedialog.askdirectory(title="Выберите папку для вывода")
    dir_entry.delete(0, tk.END)
    dir_entry.insert(0, dir_path)


def convert_file():
    file_path = file_entry.get()
    output_dir = dir_entry.get()
    if not file_path or not output_dir:
        messagebox.showerror("Ошибка", "Пожалуйста, выберите файл и папку для вывода.")
        return

    if file_path.lower().endswith(".pdf"):
        pdf_to_markdown(file_path, output_dir)
    elif file_path.lower().endswith(".docx"):
        docx_to_markdown(file_path, output_dir)
    elif file_path.lower().endswith(".txt"):
        txt_to_markdown(file_path, output_dir)
    else:
        messagebox.showerror("Ошибка", "Неподдерживаемый тип файла.")
        return
    messagebox.showinfo("Готово", "Преобразование завершено.")


# Создаем главное окно tkinter
root = tk.Tk()
root.title("Конвертер файлов в Markdown")

# Поле и кнопка выбора файла
file_label = tk.Label(root, text="Файл:")
file_label.grid(row=0, column=0, padx=5, pady=5)
file_entry = tk.Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=5, pady=5)
file_button = tk.Button(root, text="Выбрать файл", command=select_file)
file_button.grid(row=0, column=2, padx=5, pady=5)

# Поле и кнопка выбора папки
dir_label = tk.Label(root, text="Папка:")
dir_label.grid(row=1, column=0, padx=5, pady=5)
dir_entry = tk.Entry(root, width=50)
dir_entry.grid(row=1, column=1, padx=5, pady=5)
dir_button = tk.Button(root, text="Выбрать папку", command=select_directory)
dir_button.grid(row=1, column=2, padx=5, pady=5)

# Кнопка для запуска конвертации
convert_button = tk.Button(root, text="Конвертировать", command=convert_file)
convert_button.grid(row=2, column=1, padx=5, pady=10)

# Запускаем главный цикл tkinter
root.mainloop()