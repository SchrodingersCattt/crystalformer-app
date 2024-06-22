import gradio as gr
from bohrium_open_sdk import OpenSDK
from http.cookies import SimpleCookie
from typing import Tuple

def get_ak_info_from_request(request: gr.Request) -> Tuple[str, str]:
    cookie = request.headers.get("cookie")
    if cookie:
        simple_cookie = SimpleCookie()
        simple_cookie.load(cookie)
        access_key = simple_cookie["appAccessKey"].value
        app_key = simple_cookie["clientName"].value

    return access_key, app_key

def get_user_id(request: gr.Request) -> str:

    access_key, app_key = get_ak_info_from_request(request)
    client = OpenSDK(
        access_key = access_key,
        app_key = app_key
    )

    user_info = client.user.get_info()

    response = f"Get userid successfully. The user_id is: {user_info['data']['user_id']}"

    return response

with gr.Blocks() as app:
    gr.Markdown("## Getting user's ID")

    btn = gr.Button("Visualize user's ID")
    output = gr.Textbox(label="User's ID") 

    btn.click(get_user_id, outputs=output)

if __name__ == "__main__":
    app.launch(server_name='0.0.0.0', server_port=50001)

