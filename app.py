import gradio as gr
from kubernetes import client, config
from openai import OpenAI
import yaml

openAiClient = OpenAI()
config.load_kube_config()

def list_namespaces():
    """
    Retrieves a list of namespaces from the Kubernetes cluster.

    Returns:
        A list of namespace names.
    """
    v1 = client.CoreV1Api()
    namespaces = v1.list_namespace()
    return [namespace.metadata.name for namespace in namespaces.items]

def list_pods(namespace):
    """
    List all pods in the specified namespace.

    Args:
        namespace (str): The namespace to list pods from.

    Returns:
        list: A list of pod names.
    """
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    return [pod.metadata.name for pod in pods.items]

def get_pod_info(namespace, pod, include_events, include_logs):
    """
    Retrieves information about a specific pod in a given namespace.

    Args:
        namespace (str): The namespace of the pod.
        pod (str): The name of the pod.
        include_events (bool): Flag indicating whether to include events associated with the pod.
        include_logs (bool): Flag indicating whether to include logs of the pod.

    Returns:
        dict: A dictionary containing the pod information, events (if include_events is True), and logs (if include_logs is True).
    """
    v1 = client.CoreV1Api()
    pod_info = v1.read_namespaced_pod(pod, namespace)
    pod_info_map = pod_info.to_dict()
    pod_info_map["metadata"]["managed_fields"] = None
    
    info = {
        "PodInfo": pod_info_map,
    }
    if include_events:
        events = v1.list_namespaced_event(namespace)
        info["Events"] = [
            {
                "Name": event.metadata.name,
                "Message": event.message,
                "Reason": event.reason,
            }
            for event in events.items
            if event.involved_object.name == pod
        ]
    if include_logs:
        logs = v1.read_namespaced_pod_log(pod, namespace)
        info["Logs"] = logs
    
    return info


def create_prompt(msg, info):
    """
    Creates a prompt message with the given message and pod information.

    Args:
        msg (str): The main message for the prompt.
        info (dict): A dictionary containing pod information.

    Returns:
        str: The generated prompt message.
    """
    prompt = f"{msg}\n"
    prompt += f"Pod Info: \n {yaml.dump(info['PodInfo'])} \n"
    if info.get("Events"):
        prompt += f"Here are the last few events for the pod: \n"
        for event in info["Events"]:
            prompt += f"Event: {event['Name']}, Reason: {event['Reason']}, Message: {event['Message']}\n"
    if info.get("Logs"):
        prompt += f"Here are the last few logs for the pod: \n"
        prompt += info["Logs"]
    return prompt


def call_llm(msg, namespace, pod, include_events, include_logs):
    """
    Calls the Language Model to generate a response based on the given message, namespace, pod, and options.

    Args:
        msg (str): The user's message.
        namespace (str): The namespace of the pod.
        pod (str): The name of the pod.
        include_events (bool): Whether to include events in the pod information.
        include_logs (bool): Whether to include logs in the pod information.

    Returns:
        str: The generated response from the Language Model.
    """
    info = get_pod_info(namespace, pod, include_events, include_logs)
    prompt = create_prompt(msg, info)
    
    completion = openAiClient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a kubernetes expert, and will help the user with request below based on the context and info provided"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def namespace_change(value):
    """
    Returns a Dropdown widget populated with a list of pods based on the given value.

    Args:
        value (str): The namespace to filter the pods.

    Returns:
        gr.Dropdown: A Dropdown widget with choices populated from the list of pods.
    """
    pods = list_pods(value)
    return gr.Dropdown(choices=pods)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        title = gr.Label("Pod Doctor")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height="600px")
        with gr.Column(scale=1):
            namespace = gr.Dropdown(
                list_namespaces(),
                label="Namespace",
                info="Select the namespace you want to check the pod in.",
                interactive=True,
            )
            pod = gr.Dropdown(
                [],
                label="Pod",
                info="Select the pod you want to interact with.",
                interactive=True,
            )
            include_events = gr.Checkbox(
                label="Include Events",
                info="Include the pod events.",
                interactive=True,
            )
            include_logs = gr.Checkbox(
                label="Include Logs",
                info="Include the pod logs.",
                interactive=True,
            )
            namespace.change(namespace_change, inputs=namespace, outputs=pod)
    with gr.Row():
        msg = gr.Textbox()

    def respond(message, chat_history, namespace, pod, include_events, include_logs):
        if not namespace or not pod or not message:
            chat_history.append((message or "Error", "Please select a namespace and pod."))
            return "", chat_history
        bot_message = call_llm(message, namespace, pod, include_events, include_logs)
        new_message = f"Namespace {namespace}, Pod: {pod}, Include Events: {include_events}, Include Logs: {include_logs} \n {message}"
        chat_history.append((new_message, bot_message))
        return "", chat_history
    msg.submit(respond, [msg, chatbot, namespace, pod, include_events, include_logs], [msg, chatbot])


if __name__ == "__main__":
    demo.launch()