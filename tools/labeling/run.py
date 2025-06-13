from main import create_app
from config import ARGS

if __name__ == '__main__':
    app = create_app(ARGS)
    app.run(debug=False)