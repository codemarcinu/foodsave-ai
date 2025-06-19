import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import fitz  # Import biblioteki PyMuPDF
import pytesseract
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OCRResult(BaseModel):
    """Model wyniku OCR"""

    text: str
    confidence: float
    metadata: Dict[str, Any] = {}


class OCRProcessor:
    """Główna klasa do przetwarzania OCR"""

    def __init__(self, languages: List[str] = ["pol"]):
        self.languages = languages
        self.default_config = r"--oem 3 --psm 4"

    def _get_tesseract_config(self) -> str:
        """Generuje konfigurację Tesseract z uwzględnieniem języków"""
        lang_part = f"-l {'+'.join(self.languages)}" if self.languages else ""
        return f"{self.default_config} {lang_part}"

    def process_image(self, image_bytes: bytes) -> OCRResult:
        """Przetwarza obraz na tekst"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            config = self._get_tesseract_config()
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )

            text = "\n".join([line for line in data["text"] if line.strip()])
            avg_conf = (
                sum(conf for conf in data["conf"] if conf > 0) / len(data["conf"])
                if data["conf"]
                else 0
            )

            return OCRResult(
                text=text,
                confidence=avg_conf,
                metadata={
                    "source": "image",
                    "pages": 1,
                    "language": self.languages[0] if self.languages else "unknown",
                },
            )
        except Exception as e:
            logger.error(f"OCR image processing error: {e}")
            return OCRResult(text="", confidence=0, metadata={"error": str(e)})

    def process_pdf(self, pdf_bytes: bytes) -> OCRResult:
        """Przetwarza plik PDF na tekst"""
        try:
            full_text = []
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                page_result = self.process_image(image.tobytes())
                full_text.append(page_result.text)

            return OCRResult(
                text="\n".join(full_text),
                confidence=0,  # PDF confidence calculation would be more complex
                metadata={
                    "source": "pdf",
                    "pages": len(pdf_document),
                    "language": self.languages[0] if self.languages else "unknown",
                },
            )
        except Exception as e:
            logger.error(f"OCR PDF processing error: {e}")
            return OCRResult(text="", confidence=0, metadata={"error": str(e)})


def _extract_text_from_image_obj(image: Image.Image) -> str:
    """
    Prywatna funkcja pomocnicza, która wykonuje OCR na obiekcie obrazu PIL.
    """
    custom_config = r"--oem 3 --psm 4 -l pol"
    return pytesseract.image_to_string(image, config=custom_config)


def process_image_file(file_bytes: bytes) -> Optional[str]:
    """
    Przetwarza plik obrazu (jpg, png) i wyciąga z niego tekst.
    """
    try:
        logger.info("OCR: Rozpoczynam odczyt pliku obrazu...")
        image = Image.open(io.BytesIO(file_bytes))
        text = _extract_text_from_image_obj(image)
        logger.info("OCR: Odczyt obrazu zakończony sukcesem.")
        return text
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania obrazu: {e}")
        return None


def process_pdf_file(file_bytes: bytes) -> Optional[str]:
    """
    Przetwarza plik PDF, konwertując każdą stronę na obraz i odczytując tekst.
    """
    try:
        logger.info("OCR: Rozpoczynam odczyt pliku PDF...")
        full_text = []
        # Otwieramy dokument PDF z bajtów
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

        # Iterujemy po każdej stronie dokumentu
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Konwertujemy stronę na obraz (pixmap)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            # Tworzymy obiekt obrazu PIL z pixmapa
            img_size: Tuple[int, int] = (pix.width, pix.height)
            image = Image.frombytes("RGB", img_size, pix.samples)

            # Używamy naszej funkcji do odczytu tekstu z obrazu strony
            page_text = _extract_text_from_image_obj(image)
            full_text.append(page_text)

        logger.info(
            f"OCR: Odczyt PDF (stron: {len(pdf_document)}) zakończony sukcesem."
        )
        return "\n".join(full_text)

    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania PDF: {e}")
        return None
