import hopsworks
from config import DERIVED_FG_NAME

project = hopsworks.login()
fs = project.get_feature_store()

derived_fg = fs.get_feature_group(name=DERIVED_FG_NAME, version=1)
df = derived_fg.read()

print("Columns in Feature Group:", df.columns)
