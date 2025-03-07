import google.generativeai as genai
import os
api_key = os.getenv("GOOGLE_API_KEY")
# Configure your API key
genai.configure(api_key="AIzaSyCIwEqi3DiCzIOwTj71cwKWyonNslzbaIA")  # Replace with your actual API key

model = genai.GenerativeModel('gemini-2.0-flash')  # Or 'gemini-1.0-pro-vision' if you plan to use images

def generate_insights(sorrounding_data, pest_probability):
    
    prompt = (
        "You are a thermal imaging pest detection expert. I will provide you with two pieces of data: "
        "a dictionary named 'sorrounding_data' containing sensor measurements and a value 'pest_probability' "
        "indicating the likelihood of pest detection (from 0 to 1).\n\n"
        "The dictionary 'sorrounding_data' includes the following keys:\n"
        " - Thermal_Brightness\n"
        " - Glare_Level\n"
        " - Surrounding_Brightness\n"
        " - Thermal_Contrast\n"
        " - Thermal_Max\n"
        " - Thermal_Min\n"
        " - Thermal_Std\n"
        " - Ambient_Temperature\n\n"
        "This is the data on which I need insights :\n"
        f"  sorrounding_data: {sorrounding_data}\n"
        f"  pest_probability: {pest_probability}\n\n"
        "Based on these inputs, generate a detailed analysis in a paragraph of 4 to 5 sentences. "
        "Your analysis should explain how each parameter affects the quality of the thermal image and influences "
        "the pest detection outcome. Be sure to discuss the role of thermal brightness, glare level, and surrounding "
        "brightness in establishing the imaging conditions, and how thermal contrast along with maximum, minimum, and "
        "standard deviation values indicate the presence of temperature anomalies. Also, address the impact of ambient "
        "temperature on the overall detection capability. Your explanation should be clear, technically sound, and "
        "presented in a professional tone."
    )

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insight: {e}"



