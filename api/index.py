from app import app as flask_app
from vercel_wsgi import handle

def handler(request, context):
	return handle(request, flask_app)