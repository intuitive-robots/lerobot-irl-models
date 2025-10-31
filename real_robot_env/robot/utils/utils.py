import cv2


def resize_and_crop(
        cam_img_array, position, des_width: int = 500, des_height: int = 500
    ):
    """
    Resizes and crops an image to a square shape, centered or aligned to the left or right side.

    Arguments:
    - image_array: a NumPy array representing the image to be resized and cropped.
    - position: a string specifying the position of the crop within the image.
                Valid values are 'left', 'right', 'center', and 'center_right'.
    - size: a tuple specifying the desired size of the output image after resizing. The default value is (500, 500).

    Returns:
    - image_resized: a NumPy array representing the resized and cropped image.

    The method first calculates the size of the largest square that fits inside the original image,
    then crops the image to a square centered or aligned to the specified position.
    Finally, the cropped image is resized to the specified output size using the OpenCV library.

    Note that this method modifies the input image array in-place,
    so make a copy of the original image if you need to keep it intact.
    """


    # cam_image = np.load(cam_img_path)

    size = (des_width, des_height)
    # O Get the height and width of the image
    height, width, _ = cam_img_array.shape
    # Calculate the size of the square
    square_size = min(width, height)
    # Calculate the left, top, right, and bottom coordinates of the square
    if position == "left":
        # Crop the left side of the image
        left = 0
        top = 0
        right = square_size
        bottom = square_size
    elif position == "right":
        # Crop the right side of the image
        left = width - square_size
        top = 0
        right = width
        bottom = square_size
    elif position == "center":
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size 
        bottom = top + square_size 
    elif position == "top_center_new_lab":
        left = 550
        top = 0
        right = 1700
        bottom = top + square_size 
    elif position == "front_center_new_lab":
        left = 0
        top = 0
        right = 1700
        bottom = top + square_size
    elif position == "center_left":
        left = (width - square_size) // 2 - 30
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    elif position == "center_far_left":
        left = (width - square_size) // 2 - 100
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    elif position == "center_right":
        left = (width - square_size) // 2 + 100
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
    else:
        raise ValueError("Invalid position. Use 'left' or 'right' or 'center'.")

    # Crop the image to create a square
    image_cropped = cam_img_array[top:bottom, left:right]

    # Resize the image to 500x500
    image_resized = cv2.resize(image_cropped, size)

    # Downscale image to 250x250
    # image_resized = cv2.resize(image_resized, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)

    return image_resized

# Downscale image to downscale_res x downscale_res
def downscale(image_resized, downscale_res):
    return cv2.resize(image_resized, dsize=(downscale_res, downscale_res), interpolation=cv2.INTER_CUBIC)