import requests
import base64
import threading

def send_request(prompt, image_path):
    with open(image_path, "rb") as original_file:
        image_encoded_string = base64.b64encode(original_file.read()).decode("ascii")

    result = requests.post("http://localhost:8000/", json={
        "task_prompt": prompt, 
        "image": image_encoded_string,
    })


    if result.status_code == 200:
        print(result.json())
    else:
        print(f"error: {result.status_code} {result.text}")


if __name__ =="__main__":
    threads = [ threading.Thread(target=send_request, args=("<OD>", "./data/car.jpg",)),
                threading.Thread(target=send_request, args=("<CAPTION>", "./data/images_Siberian_Husky.jpg",)),
                threading.Thread(target=send_request, args=("<DETAILED_CAPTION>", "./data/car.jpg",)),
                threading.Thread(target=send_request, args=("<CAPTION>", "./data/images_Siberian_Husky.jpg",)),
                threading.Thread(target=send_request, args=("<CAPTION>", "./data/car.jpg",)),
                threading.Thread(target=send_request, args=("<OD>", "./data/images_Siberian_Husky.jpg",)),
                threading.Thread(target=send_request, args=("<OD>", "./data/car.jpg",)),
                threading.Thread(target=send_request, args=("<CAPTION>", "./data/images_Siberian_Husky.jpg",)),
              ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("Done!")
