from document_ai.schemas import BoundingBox


def normalize_bounding_box(
    bounding_box: BoundingBox, page_width: int | float, page_height: int | float
) -> BoundingBox:
    return BoundingBox(
        x0=bounding_box.x0 / page_width,
        top=bounding_box.top / page_height,
        x1=bounding_box.x1 / page_width,
        bottom=bounding_box.bottom / page_height,
    )


def denormalize_bounding_box(
    bounding_box: BoundingBox, page_width: int | float, page_height: int | float
) -> BoundingBox:
    return BoundingBox(
        x0=bounding_box.x0 * page_width,
        top=bounding_box.top * page_height,
        x1=bounding_box.x1 * page_width,
        bottom=bounding_box.bottom * page_height,
    )
