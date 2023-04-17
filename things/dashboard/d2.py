import panel as pn
from pydantic import BaseModel

class Pyfig(BaseModel):
	opt: list = ['Adam', 'SGD']
	float_array: list = []  # str to List[float]
	test: bool = True

	class App(BaseModel):
		name: str = "default_name1"
		value1: int = 0
		value2: float = 0.0

	class Model(BaseModel):
		title: str = "default_title"
		count: int = 0

	app: App = App()
	model: Model = Model()

def get_widgets(c: Pyfig or dict):
    if not isinstance(c, dict):
        c = c.dict()

    widgets = []
    for k, value in c.items():
        if isinstance(value, dict):
            widgets.append(pn.pane.Markdown(f"## {k.capitalize()}"))
            widgets.extend(get_widgets(value))
            continue

        if isinstance(value, str):
            widget_type = pn.widgets.TextInput
        elif isinstance(value, int):
            widget_type = pn.widgets.IntInput
        elif isinstance(value, float):
            widget_type = pn.widgets.FloatInput
        elif isinstance(value, bool):
            widget_type = pn.widgets.Checkbox
        elif isinstance(value, list):
            widget_type = pn.widgets.TextInput
            value = ','.join([str(v) for v in value])
        else:
            widget_type = pn.widgets.TextAreaInput

        widget = widget_type(value=value, name=k)
        widgets.append(widget)

    return widgets


def build_dashboard(c: dict):
    widgets = get_widgets(c)
    print(widgets)
    dashboard = pn.Column(*widgets)
    return dashboard, c

c = Pyfig()
print('init:d ', c.dict(), sep='\n')

dashboard, c = build_dashboard(c)

dashboard.servable()
# Display the dashboard

# pn.serve(dashboard)
