import datetime
import secrets
import string
import pandas as pd

file_path = "very very old projects\Password Generator That Stores Passwords\data.csv"


print(
    "Try to input 'open up data' for the input asking how many letters you would like!"
)


def generate_random_password(length):
    characters = string.ascii_letters + string.digits + string.punctuation
    password = "".join(secrets.choice(characters) for _ in range(length))
    return password


ask = input("How many letters would you like?")
if ask == "open up data":
    print("Sure!")
    import time

    df = pd.read_csv(file_path)
    print(df)
else:

    random_password = generate_random_password(int(ask))
    print("This is your password !")
    df = pd.read_csv(file_path)
    new_data = pd.DataFrame(
        {"Password": [random_password], "Time": [str(datetime.datetime.now())]}
    )
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(file_path, index=False)
    print(random_password)
