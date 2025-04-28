#pip install gradio openai Pillow
#python -m pip install gradio openai Pillow requests
import os
import shutil
#避免Gradio 带proxy启动失败
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
# 获取temp文件夹路径
temp_dir = os.path.join(os.getenv('LOCALAPPDATA'), 'Temp', 'gradio')
print(f"Cleaning up temp directory: {temp_dir}")
# 如果temp文件夹不存在则创建
os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None

# 启动后清空temp文件夹（删除所有内容）
for file in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, file)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete temp gradio file %s. Reason: %s' % (file_path, e))

import gradio as gr
import openai

import base64
from PIL import Image
from io import BytesIO
import warnings
import requests

# Load environment variables (especially OPENAI_API_KEY)

# --- Configuration ---
import yaml

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Failed to load config.yaml: {e}")
        return {}

config = load_config()
API_KEY = config.get("API_KEY", "")
BASE_URL = config.get("BASE_URL", "")

if not API_KEY:
    warnings.warn("API_KEY not found in config.yaml. The app will likely fail.")

# --- OpenAI Client Initialization ---
# The OpenAI client (using httpx underneath) should automatically respect
# standard proxy environment variables like HTTP_PROXY and HTTPS_PROXY.
# Ensure these are set in your system environment or .env file *before* running the script.
try:
    if BASE_URL:
        client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    else:
        client = openai.OpenAI(api_key=API_KEY)
except Exception as e:
    warnings.warn(f"Failed to initialize OpenAI client: {e}")
    client = None


 #os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    warnings.warn("OPENAI_API_KEY environment variable not found. The app will likely fail.")
    # You could alternatively raise an error here or prompt the user in the UI
    # raise ValueError("Missing OPENAI_API_KEY environment variable")



# --- Constants ---
AVAILABLE_SIZES = ["1024x1024", "1024x1536", "1536x1024","256x256", "512x512", "auto"]
AVAILABLE_QUALITIES = ["standard", "hd", "low", "medium", "high", "auto"] # Combined options
GENERATION_MODELS = ["gpt-image-1", "dall-e-3"]
EDITING_MODELS = ["gpt-image-1", "dall-e-2"]
BACKGROUND_OPTIONS = ["opaque", "transparent"] # gpt-image-1 only
OUTPUT_FORMATS = ["png", "jpeg", "webp"] # png is default, others allow compression

# --- Helper Functions ---
def decode_image(b64_string):
    """Decodes a base64 string into a PIL Image."""
    if not b64_string:
        raise ValueError("Received empty base64 string.")
    try:
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 string: {e}")


def pil_to_bytesio(pil_image, format="PNG"):
    """Converts a PIL Image to a BytesIO object."""
    img_byte_arr = BytesIO()
    save_format = format.upper()

    # Handle image modes appropriately for saving
    if save_format == "PNG":
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA') # PNG supports RGBA well
    elif save_format in ["JPEG", "WEBP"]: # Common formats not always supporting alpha
        if pil_image.mode == 'RGBA':
            # Create a white background and paste the RGBA image onto it
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3]) # 3 is the alpha channel
            pil_image = background
        elif pil_image.mode == 'P': # Palette mode, convert to RGB
             pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'LA': # Luminance + Alpha
             pil_image = pil_image.convert('RGBA').convert('RGB') # Go via RGBA then strip alpha

    pil_image.save(img_byte_arr, format=save_format)
    img_byte_arr.seek(0) # Rewind the buffer to the beginning
    return img_byte_arr

# --- API Call Functions ---

def generate_image_api(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "auto",
    output_format: str = "png",
    background: str = "opaque",
    compression: int = 75
) -> Image.Image:
    """Calls the OpenAI Image Generation API."""
    if not client:
        raise gr.Error("OpenAI client not initialized. Check API Key and Proxy settings.")
    if not prompt:
        raise gr.Error("Prompt cannot be empty.")

    # Define base parameters - response_format is NOT included here
    api_params = {
        "model": model,
        "prompt": prompt,
        "n": 1,
    }

    # --- Handle Model-Specific Parameters ---
    if model == "dall-e-3":
        if size == "auto": size = "1024x1024"
        if quality not in ["standard", "hd"]: quality = "standard"
        api_params["size"] = size
        api_params["quality"] = quality
        if background == "transparent": warnings.warn("DALL-E 3 does not support transparent backgrounds. Ignoring.")
        if output_format != "png": warnings.warn("DALL-E 3 primarily uses png. Requested format might be ignored.")

    elif model == "gpt-image-1":
        api_params["size"] = size
        api_params["quality"] = quality
        if background == "transparent":
            if output_format not in ["png", "webp"]:
                warnings.warn("Transparency requires PNG or WEBP format. Defaulting to PNG.")
                output_format = "png"
            api_params["background"] = "transparent"
        # Note: output_format and output_compression are passed directly to gpt-image-1
        api_params["output_format"] = output_format
        if output_format in ["jpeg", "webp"]:
            api_params["output_compression"] = max(0, min(100, compression)) # Clamp compression

    else:
         raise gr.Error(f"Unsupported model for generation: {model}")

    try:
        print(f"--- Generating Image ---")
        print(f"Attempting API call. Ensure system proxy (HTTP_PROXY/HTTPS_PROXY) is set if needed.")
        print(f"Parameters: {api_params}") # Print final parameters being sent
        result = client.images.generate(**api_params)
        print('Result:' + str(result))

        # Access the base64 data from the response object
        b64_json = result.data[0].b64_json
        if not b64_json:
             # Fallback or error if needed, e.g., if URL was returned instead
             # url = result.data[0].url
             raise gr.Error("API did not return base64 image data as expected.")

        print("--- Generation Complete ---")
        return decode_image(b64_json)

    except openai.APIConnectionError as e:
        raise gr.Error(f"API Connection Error: Check network, proxy settings, and OpenAI status. {e}")
    except openai.AuthenticationError as e:
        raise gr.Error(f"Authentication Error: Check your API Key. {e}")
    except openai.RateLimitError as e:
        raise gr.Error(f"Rate Limit Exceeded: {e}")
    except openai.BadRequestError as e:
         # Display the actual error message from OpenAI
         raise gr.Error(f"OpenAI Bad Request: {e}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}")


# --- API Call Functions ---

# generate_image_api 函数保持不变...

# vvv 修改这个函数 vvv
def edit_image_api(
    input_image_pil: Image.Image, # <--- 修改参数名和类型提示，更清晰
    prompt: str,
    model: str = "gpt-image-1",
    quality: str = "auto",
    output_format: str = "png",
    background: str = "opaque",
    compression: int = 75
) -> Image.Image:
    """Calls the OpenAI Image Editing API (without mask)."""
    if not client:
        raise gr.Error("OpenAI client not initialized. Check API Key and Proxy settings.")
    if not prompt:
        raise gr.Error("Prompt cannot be empty.")
    # vvv 修改输入检查 vvv
    if not input_image_pil:
        raise gr.Error("Input image is required for editing.")

    # 直接使用传入的PIL对象
    # image_pil = input_image_dict['image'] # <-- 移除这行
    # mask_pil = input_image_dict.get('mask') # <-- 移除这行

    # 准备图像字节 (PNG preserves transparency if input has it)
    image_bytes = pil_to_bytesio(input_image_pil, format="PNG")

    # --- 移除Mask处理逻辑 ---
    # mask_bytes = None
    # if mask_pil:
    #     # ... (相关检查和转换代码全部移除) ...
    #     print("Mask provided and prepared.")
    # else:
    #     print("No mask provided.") # 可以保留这句，说明现在总是不提供mask
    print("Mask functionality disabled. Performing full image edit based on prompt.")
    # --- 结束移除Mask处理逻辑 ---


    # 定义基础API参数 - 不包含 mask
    api_params = {
        "model": model,
        "image": image_bytes,
        "prompt": prompt,
        "n": 1,
    }
    # if mask_bytes: # <-- 移除这个条件判断
    #     api_params["mask"] = mask_bytes

    # --- 模型特定参数处理 (保持不变) ---
    if model == "gpt-image-1":
        api_params["quality"] = quality
        if background == "transparent":
             if output_format not in ["png", "webp"]:
                 warnings.warn("Transparency requires PNG or WEBP. Defaulting to PNG.")
                 output_format = "png"
             api_params["background"] = "transparent"
        api_params["output_format"] = output_format
        if output_format in ["jpeg", "webp"]:
            api_params["output_compression"] = max(0, min(100, compression))
    elif model == "dall-e-2":
        raise gr.Error(f"Unsupported model for editing: {model}")
    else:
        raise gr.Error(f"Unsupported model for editing: {model}")

    # --- API 调用和结果处理 (保持不变, 包括URL下载逻辑) ---
    try:
        print(f"--- Editing Image (No Mask) ---")
        print(f"Attempting API call. Ensure system proxy (HTTP_PROXY/HTTPS_PROXY) is set if needed.")
        # 打印参数时不再需要排除 image/mask，因为 mask 已经没了
        print(f"Parameters (excluding image bytes): { {k:v for k,v in api_params.items() if k != 'image'} }")
        #不支持output_format
        del api_params["output_format"]
        #api_params['response_format'] = "b64_json"
        result = client.images.edit(**api_params)

        if not result.data:
             raise gr.Error("API edit response did not contain image data.")

        image_data = result.data[0]
        b64_json = image_data.b64_json
        image_url = image_data.url

        if b64_json:
            print("--- Editing Complete (Base64 received) ---")
            return decode_image(b64_json)
        # DALL-E 2 编辑也可能返回 URL，保留下载逻辑
        elif image_url: # Simplied check now - if URL exists, try downloading
            print(f"--- Base64 not found for edit, downloading from URL: {image_url} ---")
            try:
                proxies = { "http": os.environ.get('HTTP_PROXY'), "https": os.environ.get('HTTPS_PROXY') }
                proxies = {k: v for k, v in proxies.items() if v}
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(image_url, timeout=45, headers=headers, proxies=proxies if proxies else None)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                print("--- Editing Complete (Downloaded from URL) ---")
                return img
            except requests.exceptions.RequestException as req_err:
                 print(f"ERROR downloading edited image from URL ({image_url}): {req_err}")
                 raise gr.Error(f"Failed to download edited image from URL. Check network/proxy. Error: {req_err}")
            except Exception as decode_err:
                 print(f"ERROR processing downloaded edited image from URL ({image_url}): {decode_err}")
                 raise gr.Error(f"Failed to process downloaded edited image. Error: {decode_err}")
        else:
             print(f"ERROR: Edit API response contained neither b64_json nor url. Response data: {image_data}")
             raise gr.Error("Edit API response did not contain expected image data.")

    # --- 异常处理 (保持不变) ---
    except openai.APIConnectionError as e:
        raise gr.Error(f"API Connection Error: Check network, proxy settings, OpenAI status. {e}")
    except openai.AuthenticationError as e:
        raise gr.Error(f"Authentication Error: Check your API Key. {e}")
    except openai.RateLimitError as e:
        raise gr.Error(f"Rate Limit Exceeded: {e}")
    except openai.BadRequestError as e:
         raise gr.Error(f"OpenAI Bad Request: {e}")
    except Exception as e:
        print(f"ERROR in edit_image_api: {type(e).__name__} - {e}")
        if isinstance(e, gr.Error): raise e
        else: raise gr.Error(f"An unexpected error occurred: {e}")


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# OpenAI Image Generation & Editing by FORSAKENSILVER")
    gr.Markdown(
        """
        Enter prompts and parameters to generate or edit images.
        **Proxy Configuration:**
        *   This app attempts to use system proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`) for OpenAI API calls. Set these environment variables **before** launching the script if you need a proxy.
        *   The Gradio interface is launched on `0.0.0.0` . Access it via your machine's network IP address (e.g., http://192.168.1.100:7860).
        **API Key:** Ensure your `API_KEY` is set in the  `config.yaml` file.
        """
    )

    with gr.Tabs():
        # --- Generation Tab ---
        with gr.TabItem("Generate Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(lines=3, label="Prompt", placeholder="e.g., A photorealistic cat astronaut riding a unicorn on the moon")
                    gen_model = gr.Dropdown(GENERATION_MODELS, label="Model", value="gpt-image-1")
                    gen_size = gr.Dropdown(AVAILABLE_SIZES, label="Size", value="1024x1024")
                    gen_quality = gr.Dropdown(AVAILABLE_QUALITIES, label="Quality", value="auto")
                    gen_background = gr.Radio(BACKGROUND_OPTIONS, label="Background (gpt-image-1 only)", value="opaque")
                    with gr.Accordion("Advanced Format Options (gpt-image-1)", open=False):
                         gen_output_format = gr.Dropdown(OUTPUT_FORMATS, label="Output Format", value="png")
                         gen_compression = gr.Slider(0, 100, value=75, step=1, label="Compression (JPEG/WEBP)", visible=False)

                         def toggle_compression_visibility(fmt): return gr.update(visible=fmt in ["jpeg", "webp"])
                         gen_output_format.change(toggle_compression_visibility, inputs=gen_output_format, outputs=gen_compression)

                    gen_button = gr.Button("Generate Image", variant="primary")
                with gr.Column(scale=1):
                    gen_output_image = gr.Image(label="Generated Image", type="pil", interactive=False, format='png')

        # --- Editing Tab ---
        with gr.TabItem("Edit Image"):
            gr.Markdown("Upload an image.")
            with gr.Row():
                with gr.Column(scale=2):
                    edit_input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        sources=["upload"],
                    )
                    edit_prompt = gr.Textbox(lines=3, label="Editing Prompt", placeholder="e.g., Add sunglasses to the cat, Make the background a sunny beach")
                    edit_model = gr.Dropdown(EDITING_MODELS, label="Model", value="gpt-image-1")
                    edit_quality = gr.Dropdown(AVAILABLE_QUALITIES, label="Quality (gpt-image-1 recommended)", value="auto")
                    edit_background = gr.Radio(BACKGROUND_OPTIONS, label="Background (gpt-image-1 only)", value="opaque")
                    with gr.Accordion("Advanced Format Options (gpt-image-1)", open=False):
                         edit_output_format = gr.Dropdown(OUTPUT_FORMATS, label="Output Format", value="png")
                         edit_compression = gr.Slider(0, 100, value=75, step=1, label="Compression (JPEG/WEBP)", visible=False)

                         def toggle_compression_visibility_edit(fmt): return gr.update(visible=fmt in ["jpeg", "webp"])
                         edit_output_format.change(toggle_compression_visibility_edit, inputs=edit_output_format, outputs=edit_compression)

                    edit_button = gr.Button("Edit Image", variant="primary")
                with gr.Column(scale=1):
                    edit_output_image = gr.Image(label="Edited Image", type="pil", interactive=False, format='png')

    # --- Connect Functions to UI ---
    gen_button.click(
        fn=generate_image_api,
        inputs=[gen_prompt, gen_model, gen_size, gen_quality, gen_output_format, gen_background, gen_compression],
        outputs=gen_output_image,
        api_name="generate_image"
    )

    edit_button.click(
        fn=edit_image_api,
        inputs=[edit_input_image, edit_prompt, edit_model, edit_quality, edit_output_format, edit_background, edit_compression],
        outputs=edit_output_image,
        api_name="edit_image"
    )
# --- Launch the App ---
if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set the environment variable and restart the application.")
        # Optionally prevent app launch if key is missing
        # exit()
    elif not client:
         print("ERROR: Failed to initialize OpenAI Client. Cannot launch.")
         # exit()

    app.launch(debug=True, server_name='0.0.0.0', share=False) # debug=True provides more detailed errors
