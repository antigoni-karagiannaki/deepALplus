import os
from PIL import Image

def check_image_features(directory):
    """Checks for common and unique features of images in the specified directory.

    Args:
        directory (str): The path to the directory containing the images.

    Returns:
        dict: A dictionary with common features and a list of unique features for each image.
    """

    common_features = {
        'dimensions': None,
        'mode': None,
        'format': None
    }
    unique_features = []

    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath)
            width, height = img.size
            mode = img.mode
            img_format = img.format

            # Initialize common features if not set
            if common_features['dimensions'] is None:
                common_features['dimensions'] = (width, height)
                print(width, height)
            if common_features['mode'] is None:
                common_features['mode'] = mode
                print(mode)
            if common_features['format'] is None:
                common_features['format'] = img_format
                print(img_format)
            # Check for unique features
            if (width, height) != common_features['dimensions'] or mode != common_features['mode'] or img_format != common_features['format']:
                unique_features.append({
                    'filename': filename,
                    'dimensions': (width, height),
                    'mode': mode,
                    'format': img_format
                })
        else:
            # check if filename is a directory
            if os.path.isdir(os.path.join(directory, filename)):   
                print(f"Checking images in {filename} directory") 
                check_image_features(os.path.join(directory, filename))

    return common_features, unique_features

# Example usage:
directory_path = "data/UCMerced_LandUse/Images/UCMerced_LandUse/Images"
common_features, unique_features = check_image_features(directory_path)

print("Common features:")
print(f"Dimensions: {common_features['dimensions']}")
print(f"Mode: {common_features['mode']}")
print(f"Format: {common_features['format']}")

if unique_features:
    print("\nImages with unique features:")
    for feature in unique_features:
        print(f"Filename: {feature['filename']}")
        print(f"  Dimensions: {feature['dimensions']}")
        print(f"  Mode: {feature['mode']}")
        print(f"  Format: {feature['format']}")
else:
    print("\nAll images share the common features.")