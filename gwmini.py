from PIL import Image
import numpy as np

def estimate_temperature_from_thermal_image(image_path, min_temp_c=0, max_temp_c=100):
    """
    Estimates approximate temperature from a thermal image based on pixel intensity.

    Args:
        image_path (str): Path to the thermal image file.
        min_temp_c (int): Minimum temperature in Celsius for the range.
        max_temp_c (int): Maximum temperature in Celsius for the range.

    Returns:
        tuple: A tuple containing:
            - average_temperature_c (float): Approximate average temperature in Celsius.
            - min_max_temperature_range (tuple): Approximate minimum and maximum temperature in Celsius in the image.
            - pixel_intensity_stats (tuple): Minimum, maximum, and average pixel intensity.
    """
    try:
        img = Image.open(image_path)
        # Convert to grayscale if it's not already
        img = img.convert('L')  # 'L' mode is grayscale
        pixel_data = np.array(img)

        min_pixel_intensity = np.min(pixel_data)
        max_pixel_intensity = np.max(pixel_data)
        average_pixel_intensity = np.mean(pixel_data)

        # Simplified linear mapping from pixel intensity to temperature
        pixel_range = max_pixel_intensity - min_pixel_intensity
        if pixel_range == 0:
            average_temperature_c = (min_temp_c + max_temp_c) / 2  # Avoid division by zero if image is uniform color
            min_temperature_c = average_temperature_c
            max_temperature_c = average_temperature_c
        else:
            average_temperature_c = min_temp_c + (average_pixel_intensity - min_pixel_intensity) * (max_temp_c - min_temp_c) / pixel_range
            min_temperature_c = min_temp_c + (min_pixel_intensity - min_pixel_intensity) * (max_temp_c - min_temp_c) / pixel_range  # Will always be min_temp_c
            max_temperature_c = min_temp_c + (max_pixel_intensity - min_pixel_intensity) * (max_temp_c - min_temp_c) / pixel_range  # Will always be max_temp_c

        return average_temperature_c, (min_temperature_c, max_temperature_c), (min_pixel_intensity, max_pixel_intensity, average_pixel_intensity)

    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None


if __name__ == "__main__":
    image_file = r'/Users/sanjay/Desktop/hrbot/minor/thermal_image.jpg'
    if not image_file:
        print("No image path provided. Exiting.")
    else:
        average_temp, temp_range, pixel_stats = estimate_temperature_from_thermal_image(image_file)

        if average_temp is not None:
            print("\nTemperature Estimation Results:")
            print(f"  Approximate Average Temperature: {average_temp:.2f} °C")
            print(f"  Approximate Temperature Range in Image: {temp_range[0]:.2f} °C to {temp_range[1]:.2f} °C")
            print(f"  Pixel Intensity Statistics:")
            print(f"    Minimum Pixel Intensity: {pixel_stats[0]}")
            print(f"    Maximum Pixel Intensity: {pixel_stats[1]}")
            print(f"    Average Pixel Intensity: {pixel_stats[2]:.2f}")
        else:
            print("Could not process the image. Please check the file path and image format.")