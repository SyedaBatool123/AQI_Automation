''' import hopsworks

# Step 1: Login with API Key
project = hopsworks.login(
    api_key_value="AKTqJb9vBg5qyRHC.M7kEyD3EqY029ZjlDDZwBh1mRUqz0GDXGKt8nzLR9QdjfXKmzsbB8Yf8sdHw1MTK",
    project="syedabatool"
)

# Step 2: Get Feature Store
fs = project.get_feature_store()
print("Connected successfully to Hopsworks Feature Store!")
'''

print("Connected successfully to Hopsworks Feature Store!")


import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()  # ye .env file ko load karega

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT")
)

fs = project.get_feature_store()
print("✅ Connected successfully to Hopsworks Feature Store!")
