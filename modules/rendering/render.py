import numpy as np
import imkit as imk
from typing import Tuple, List

from PIL import Image, ImageFont, ImageDraw
from PySide6.QtGui import QColor, QFont, QTextDocument,\
      QTextCursor, QTextBlockFormat, QTextOption
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .hyphen_textwrap import wrap as hyphen_wrap
from modules.utils.textblock import TextBlock
from modules.utils.textblock import adjust_blks_size
from modules.detection.utils.geometry import shrink_bbox
from app.ui.canvas.text.vertical_layout import VerticalTextDocumentLayout
from modules.utils.language_utils import get_language_code, is_no_space_lang

from dataclasses import dataclass

SFX_SMALL_OUTLINE_WIDTH = 1.0
SFX_LARGE_OUTLINE_WIDTH = 2.0
SFX_LARGE_FONT_THRESHOLD = 32

@dataclass
class TextRenderingSettings:
    alignment_id: int
    font_family: str
    min_font_size: int
    max_font_size: int
    color: str
    upper_case: bool
    outline: bool
    outline_color: str
    outline_width: str
    bold: bool
    italic: bool
    underline: bool
    line_spacing: str
    direction: Qt.LayoutDirection

def array_to_pil(rgb_image: np.ndarray):
    # Image is already in RGB format, just convert to PIL
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_array(pil_image: Image):
    # Convert the PIL image to a numpy array (already in RGB)
    numpy_image = np.array(pil_image)
    return numpy_image

def is_vertical_language_code(lang_code: str | None) -> bool:
    """Return True if the language code should use vertical layout.

    Currently treats Japanese and simplified/traditional Chinese as
    vertical-capable languages.
    """
    if not lang_code:
        return False
    code = lang_code.lower()
    return code in {"zh-cn", "zh-tw", "ja"}

def is_vertical_block(blk, lang_code: str | None) -> bool:
    """Return True if this block should be rendered vertically.

    A block is considered vertical when its direction flag is "vertical"
    and the target language code is one of the vertical-capable ones.
    """
    return getattr(blk, "direction", "") == "vertical" and is_vertical_language_code(lang_code)

def is_sfx_block(blk) -> bool:
    profile = getattr(blk, "bubble_cleaning_profile", None) or getattr(blk, "cleaning_profile", None)
    if profile == "sfx":
        return True
    if profile == "free_text":
        return False
    return getattr(blk, "text_class", None) == "text_free"


def _box_area(box) -> float:
    if box is None or len(box) < 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _is_large_balloon(blk) -> bool:
    text_area = max(1.0, _box_area(getattr(blk, "xyxy", None)))
    bubble = getattr(blk, "bubble_xyxy", None)
    bubble_area = _box_area(bubble)
    if bubble_area <= 0:
        return False
    return bubble_area / text_area >= 4.5


def _is_dashed_balloon(blk, image) -> bool:
    """Detect a fragmented/dashed border without changing detection output."""
    bubble = getattr(blk, "bubble_xyxy", None)
    if image is None or not isinstance(image, np.ndarray) or bubble is None:
        return False
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bubble[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return False
    crop = image[y1:y2, x1:x2]
    gray = (
        np.dot(crop[..., :3], np.array([0.299, 0.587, 0.114]))
        if crop.ndim == 3 else crop
    )
    dark = (gray < 80).astype(np.uint8)
    band_y = max(3, int(dark.shape[0] * 0.22))
    band_x = max(3, int(dark.shape[1] * 0.22))
    ring = np.zeros_like(dark)
    ring[:band_y, :] = 1
    ring[-band_y:, :] = 1
    ring[:, :band_x] = 1
    ring[:, -band_x:] = 1
    border_ink = dark * ring
    count, _labels, stats, _ = imk.connected_components_with_stats(border_ink, connectivity=8)
    areas = [
        int(stats[i, imk.CC_STAT_AREA])
        for i in range(1, count)
        if stats[i, imk.CC_STAT_AREA] >= 3
    ]
    total = sum(areas)
    if len(areas) < 6 or total == 0:
        return False
    return max(areas) / float(total) < 0.48


def _is_starburst_balloon(blk, image) -> bool:
    """Detect an oval bubble ringed by pointed black spikes (shout/reveal burst)."""
    bubble = getattr(blk, "bubble_xyxy", None)
    if image is None or not isinstance(image, np.ndarray) or bubble is None:
        return False
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = [int(round(float(v))) for v in bubble[:4]]
    bw = bx2 - bx1
    bh = by2 - by1
    if bw < 24 or bh < 24:
        return False

    # Look at a donut-shaped region just outside the bubble oval, where spikes live.
    margin_x = max(6, int(bw * 0.35))
    margin_y = max(6, int(bh * 0.35))
    ox1, oy1 = max(0, bx1 - margin_x), max(0, by1 - margin_y)
    ox2, oy2 = min(w, bx2 + margin_x), min(h, by2 + margin_y)
    if ox2 - ox1 < 24 or oy2 - oy1 < 24:
        return False

    outer_crop = image[oy1:oy2, ox1:ox2]
    gray = (
        np.dot(outer_crop[..., :3], np.array([0.299, 0.587, 0.114]))
        if outer_crop.ndim == 3 else outer_crop
    )

    # Mask out the inner oval itself so we only look at the spike donut area.
    donut = np.ones(gray.shape, dtype=np.uint8)
    inner_x1 = bx1 - ox1
    inner_y1 = by1 - oy1
    inner_x2 = bx2 - ox1
    inner_y2 = by2 - oy1
    donut[max(0, inner_y1):max(0, inner_y2), max(0, inner_x1):max(0, inner_x2)] = 0

    dark = ((gray < 80).astype(np.uint8)) * donut
    if not np.any(dark):
        return False

    count, _labels, stats, _ = imk.connected_components_with_stats(dark, connectivity=8)
    spike_components = [
        int(stats[i, imk.CC_STAT_AREA])
        for i in range(1, count)
        if stats[i, imk.CC_STAT_AREA] >= 6
    ]
    # A starburst halo is made of many small, separate pointed shapes rather
    # than one or two large solid blobs.
    if len(spike_components) < 8:
        return False
    donut_area = max(1, int(np.count_nonzero(donut)))
    largest_share = max(spike_components) / float(sum(spike_components))
    dark_coverage = sum(spike_components) / float(donut_area)
    return largest_share < 0.25 and 0.08 <= dark_coverage <= 0.75


def get_render_font_style_for_block(blk, image=None, default_bold=False, default_italic=False):
    """Return (role, bold, italic) while preserving explicit block overrides."""
    explicit = getattr(blk, "render_role", None)
    if explicit in {"regular", "dotted", "large", "sfx", "starburst"}:
        role = explicit
    elif is_sfx_block(blk):
        role = "sfx"
    elif _is_starburst_balloon(blk, image):
        role = "starburst"
    elif _is_dashed_balloon(blk, image):
        role = "dotted"
    elif _is_large_balloon(blk):
        role = "large"
    else:
        role = "regular"

    blk.render_role = role
    return role, default_bold, default_italic


def get_sfx_outline_width(font_size) -> float:
    return SFX_LARGE_OUTLINE_WIDTH if float(font_size or 0) >= SFX_LARGE_FONT_THRESHOLD else SFX_SMALL_OUTLINE_WIDTH


def get_render_outline_for_block(blk, outline_color, outline_width: float):
    if is_sfx_block(blk):
        font_size = getattr(blk, "_render_font_size", 0)
        return outline_color or QColor("#ffffff"), get_sfx_outline_width(font_size), True
    return outline_color, outline_width, outline_color is not None

def _split_at_fitting_hyphen(
    current_line: str,
    word: str,
    measure_side,
    max_side: float,
) -> Tuple[str, str] | None:
    """Return the longest hyphen-preserving split that fits, if any."""

    best_split = None
    for idx, char in enumerate(word):
        if char != "-" or idx <= 0 or idx >= len(word) - 1:
            continue
        prefix = word[: idx + 1]
        candidate = prefix if not current_line else f"{current_line} {prefix}"
        if measure_side(candidate) <= max_side:
            best_split = (prefix, word[idx + 1 :])
    return best_split

def _wrap_text_greedily(text: str, measure_side, max_side: float) -> str:
    """Greedy wrapping that only splits inside words at existing hyphens."""

    words = text.split()
    lines: List[str] = []

    while words:
        line = ""
        while words:
            next_word = words[0]
            candidate = next_word if not line else f"{line} {next_word}"
            if measure_side(candidate) <= max_side:
                line = candidate
                words.pop(0)
                continue

            hyphen_split = _split_at_fitting_hyphen(line, next_word, measure_side, max_side)
            if hyphen_split is not None:
                prefix, suffix = hyphen_split
                line = prefix if not line else f"{line} {prefix}"
                words[0] = suffix
                break

            if line:
                break

            line = words.pop(0)
            break

        lines.append(line)

    return "\n".join(lines)

def _wrap_no_space_text_greedily(text: str, measure_side, max_side: float) -> str:
    """Greedy wrapping for languages that do not rely on spaces between words."""

    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    wrapped_paragraphs: List[str] = []

    for paragraph in paragraphs:
        chars = [char for char in paragraph if char != " "]
        if not chars:
            wrapped_paragraphs.append("")
            continue

        lines: List[str] = []
        line = ""

        for char in chars:
            candidate = f"{line}{char}"
            if not line or measure_side(candidate) <= max_side:
                line = candidate
                continue

            lines.append(line)
            line = char

        if line:
            lines.append(line)

        wrapped_paragraphs.append("\n".join(lines))

    return "\n".join(wrapped_paragraphs)

def pil_word_wrap(image: Image, tbbox_top_left: Tuple, font_pth: str, text: str, 
                  roi_width, roi_height, align: str, spacing, init_font_size: int, min_font_size: int = 10):
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)

    def eval_metrics(txt, font):
        """Quick helper function to calculate width/height of text."""
        (left, top, right, bottom) = ImageDraw.Draw(image).multiline_textbbox(xy=tbbox_top_left, text=txt, font=font, align=align, spacing=spacing)
        return (right-left, bottom-top)

    while font_size > min_font_size:
        font = font.font_variant(size=font_size)
        width, height = eval_metrics(mutable_message, font)
        if height > roi_height:
            font_size -= 0.75  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                if columns == 0:
                    break
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
                wrapped_width, _ = eval_metrics(mutable_message, font)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 0.75  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break

    if font_size <= min_font_size:
        font_size = min_font_size
        mutable_message = text
        font = font.font_variant(size=font_size)

        # Wrap text to fit within as much as possible
        # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
        # This is a brute force approach, but it works well enough
        min_cost = 1e9
        min_text = text
        for columns in range(1, len(text)):
            wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
            wrapped_width, wrapped_height = eval_metrics(wrapped_text, font)
            cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
            if cost < min_cost:
                min_cost = cost
                min_text = wrapped_text

        mutable_message = min_text

    return mutable_message, font_size

def draw_text(image: np.ndarray, blk_list: List[TextBlock], font_pth: str, colour: str = "#000", init_font_size: int = 40, min_font_size=10, outline: bool = True):
    image = array_to_pil(image)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_pth, size=init_font_size)

    for blk in blk_list:
        x1, y1, width, height = blk.xywh
        tbbox_top_left = (x1, y1)

        translation = blk.translation
        if not translation or len(translation) == 1:
            continue

        if blk.min_font_size > 0:
            min_font_size = blk.min_font_size
        if blk.max_font_size > 0:
            init_font_size = blk.max_font_size
        if blk.font_color:
            colour = blk.font_color

        translation, font_size = pil_word_wrap(image, tbbox_top_left, font_pth, translation, width, height,
                                               align=blk.alignment, spacing=blk.line_spacing, init_font_size=init_font_size, min_font_size=min_font_size)
        font = font.font_variant(size=font_size)

        # Font Detection Workaround. Draws white color offset around text
        if outline:
            offsets = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2) if dx != 0 or dy != 0]
            for dx, dy in offsets:
                draw.multiline_text((tbbox_top_left[0] + dx, tbbox_top_left[1] + dy), translation, font=font, fill="#FFF", align=blk.alignment, spacing=1)
        draw.multiline_text(tbbox_top_left, translation, colour, font, align=blk.alignment, spacing=1)
        
    image = pil_to_array(image)  # Already in RGB format
    return image

def get_best_render_area(blk_list: List[TextBlock], img, inpainted_img=None):
    # Using Speech Bubble detection to find best Text Render Area
    
    # if inpainted_img is None or inpainted_img.size == 0:
    #     return blk_list
    
    for blk in blk_list:
        if blk.text_class == 'text_bubble' and blk.bubble_xyxy is not None:
            
            if blk.source_lang_direction == 'vertical':
                text_draw_bounds = shrink_bbox(blk.bubble_xyxy, shrink_percent=0.3)
                bdx1, bdy1, bdx2, bdy2 = text_draw_bounds
                blk.xyxy[:] = [bdx1, bdy1, bdx2, bdy2]

    if blk_list and blk_list[0].source_lang not in ['ko', 'zh']:
        adjust_blks_size(blk_list, img, -5, -5)

    return blk_list


def pyside_word_wrap(
    text: str, 
    font_input: str, 
    roi_width: int, 
    roi_height: int,
    line_spacing: float, 
    outline_width: float, 
    bold: bool, 
    italic: bool, 
    underline: bool, 
    alignment: Qt.AlignmentFlag,
    direction: Qt.LayoutDirection, 
    init_font_size: int, 
    min_font_size: int = 10, 
    vertical: bool = False,
    no_space_language: bool = False,
    return_metrics: bool = False
) -> tuple:
    
    """Break long text to multiple lines, and find the largest point size
        so that all wrapped text fits within the box."""
    
    def prepare_font(font_size):
        effective_family = font_input.strip() if isinstance(font_input, str) and font_input.strip() else QApplication.font().family()
        font = QFont(effective_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)

        return font

    def eval_metrics(
        txt: str,
        font_sz: float,
        vertical: bool = False,
        include_outline: bool = True
    ) -> Tuple[float, float]:
        """Quick helper function to calculate width/height of text using QTextDocument."""
        
        # Create a QTextDocument
        doc = QTextDocument()
        doc.setDefaultFont(prepare_font(font_sz))
        doc.setPlainText(txt)

        # Set text direction
        text_option = QTextOption()
        text_option.setTextDirection(direction)
        doc.setDefaultTextOption(text_option)

        if vertical:
            layout = VerticalTextDocumentLayout(
                document=doc,
                line_spacing=line_spacing
            )

            doc.setDocumentLayout(layout)
            layout.update_layout()
        else:
            # Apply line spacing
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.SelectionType.Document)
            block_format = QTextBlockFormat()
            spacing = line_spacing * 100
            block_format.setLineHeight(spacing, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            block_format.setAlignment(alignment)
            cursor.mergeBlockFormat(block_format)
        
        # Get the size of the document
        size = doc.size()
        width, height = size.width(), size.height()
        
        # Add outline width to the size
        if include_outline and outline_width > 0:
            width += 2 * outline_width
            height += 2 * outline_width
        
        return width, height

    def wrap_and_size(font_size):
        def measure_side(candidate: str) -> float:
            w, h = eval_metrics(candidate, font_size, vertical)
            return h if vertical else w

        if no_space_language:
            wrapped = _wrap_no_space_text_greedily(
                text=text,
                measure_side=measure_side,
                max_side=roi_height if vertical else roi_width,
            )
        else:
            wrapped = _wrap_text_greedily(
                text=text,
                measure_side=measure_side,
                max_side=roi_height if vertical else roi_width,
            )
        # measure wrapped block
        w, h = eval_metrics(wrapped, font_size, vertical)
        return wrapped, w, h
    
    # Initialize
    best_text, best_size = text, init_font_size
    found_fit = False

    lo, hi = min_font_size, init_font_size
    while lo <= hi:
        mid = (lo + hi) // 2
        wrapped, w, h = wrap_and_size(mid)
        if w <= roi_width and h <= roi_height:
            found_fit = True
            best_text, best_size = wrapped, mid
            lo = mid + 1
        else:
            hi = mid - 1

    # if nothing ever fit, force a wrap at the minimum size
    if not found_fit:
        best_text, w, h = wrap_and_size(min_font_size)
        best_size = min_font_size

    if return_metrics:
        # Match persisted state to the text item's actual geometry.
        rendered_w, rendered_h = eval_metrics(best_text, best_size, vertical, include_outline=False)
        return best_text, best_size, rendered_w, rendered_h

    return best_text, best_size

    # mutable_message = text
    # font_size = init_font_size
    # # font_size = max(roi_width, roi_height)

    # while font_size > min_font_size:
    #     width, height = eval_metrics(mutable_message, font_size)
    #     if height > roi_height:
    #         font_size -= 1  # Reduce pointsize
    #         mutable_message = text  # Restore original text
    #     elif width > roi_width:
    #         columns = len(mutable_message)
    #         while columns > 0:
    #             columns -= 1
    #             if columns == 0:
    #                 break
    #             mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
    #             wrapped_width, _ = eval_metrics(mutable_message, font_size)
    #             if wrapped_width <= roi_width:
    #                 break
    #         if columns < 1:
    #             font_size -= 1  # Reduce pointsize
    #             mutable_message = text  # Restore original text
    #     else:
    #         break

    # if font_size <= min_font_size:
    #     font_size = min_font_size
    #     mutable_message = text

    #     # Wrap text to fit within as much as possible
    #     # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
    #     min_cost = 1e9
    #     min_text = text
    #     for columns in range(1, len(text)):
    #         wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
    #         wrapped_width, wrapped_height = eval_metrics(wrapped_text, font_size)
    #         cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
    #         if cost < min_cost:
    #             min_cost = cost
    #             min_text = wrapped_text

    #     mutable_message = min_text

    # return mutable_message, font_size

def manual_wrap(
    main_page, 
    blk_list: List[TextBlock], 
    image_path: str,
    font_family: str, 
    line_spacing: float, 
    outline_width: float, 
    bold: bool, 
    italic: bool, 
    underline: bool, 
    alignment: Qt.AlignmentFlag, 
    direction: Qt.LayoutDirection, 
    init_font_size: int = 40, 
    min_font_size: int = 10
):
    
    target_lang = main_page.lang_mapping.get(main_page.t_combo.currentText(), None)
    trg_lng_cd = get_language_code(target_lang)

    detection_image = None
    try:
        detection_image = main_page.image_viewer.get_image_array()
    except Exception:
        detection_image = None

    for blk in blk_list:
        x1, y1, width, height = blk.xywh

        translation = blk.translation
        if not translation or len(translation) == 1:
            continue

        vertical = is_vertical_block(blk, trg_lng_cd)
        role, block_bold, block_italic = get_render_font_style_for_block(
            blk, detection_image, bold, italic
        )
        _outline_color, effective_outline_width, _outline_enabled = get_render_outline_for_block(
            blk,
            QColor("#ffffff"),
            outline_width,
        )

        translation, font_size = pyside_word_wrap(
            translation, 
            font_family, 
            width, 
            height,
            line_spacing, 
            effective_outline_width, 
            block_bold,
            block_italic,
            underline,
            alignment, 
            direction, 
            init_font_size, 
            min_font_size,
            vertical,
            is_no_space_lang(trg_lng_cd)
        )
        blk._render_font_size = font_size
        if role == "sfx":
            _outline_color, resolved_width, _outline_enabled = get_render_outline_for_block(
                blk, QColor("#ffffff"), outline_width
            )
            if resolved_width != effective_outline_width:
                translation, font_size = pyside_word_wrap(
                    blk.translation, font_family, width, height, line_spacing,
                    resolved_width, block_bold, block_italic, underline,
                    alignment, direction, init_font_size, min_font_size,
                    vertical, is_no_space_lang(trg_lng_cd)
                )
                blk._render_font_size = font_size
        
        main_page.blk_rendered.emit(translation, font_size, blk, image_path)



        
