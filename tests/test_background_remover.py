from io import BytesIO

from PIL import Image

from src.shared.background_remover import compose_cutout, image_to_png_bytes


def test_compose_cutout_applies_resized_alpha_mask() -> None:
    image = Image.new("RGB", (4, 2), "red")
    mask = Image.new("L", (2, 1))
    mask.putdata([0, 255])

    result = compose_cutout(image, mask)

    assert result.mode == "RGBA"
    assert result.size == image.size
    assert result.getchannel("A").getpixel((0, 0)) == 0
    assert result.getchannel("A").getpixel((3, 0)) == 255


def test_image_to_png_bytes_preserves_transparency() -> None:
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 0))

    png_bytes = image_to_png_bytes(image)
    reopened = Image.open(BytesIO(png_bytes))

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert reopened.mode == "RGBA"
    assert reopened.getpixel((0, 0))[3] == 0
