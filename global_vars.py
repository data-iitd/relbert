global path
global db
global sql_path
global dict_path
global model_path

path = "/home/garima/relevant_files/"
db = "imdb"
sql_path = path + db +"/sql_files/"
dict_path = path +db +"/dict/"
model_path = path + db + "/model/"

def get_path():
	return path

def get_dict_path():
	return dict_path

def get_model_path():
	return model_path

def get_sql_path():
	return sql_path
