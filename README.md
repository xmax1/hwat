





# How To
- Fill in User Details in Pyfig



# Doc Notes
## Glossary







# Gist / Notion / Embed / Share
- https://blog.shorouk.dev/2020/06/how-to-embed-any-number-of-html-widgets-snippets-into-notion-app-for-free/

https://hwat.herokuapp.com/panel_demo


# Setup
## Requirements file
- pipreqsnb <jup-notebook>
- pipreqs <python-file>

## Procfile
### Abstract
- <indicate-what-kind-of-app-as-defined-by-heroku>: <a-cmd-to-run-the-app>
### Generalisation
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com
### Description
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com
### Example
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com

# Heroku
## tar install (NO MARCHA)
"""
wget https://cli-assets.heroku.com/branches/stable/heroku-OS-ARCH.tar.gz
tar -xvzf heroku-OS-ARCH -C /usr/local/lib/heroku
ln -s /usr/local/lib/heroku/bin/heroku /usr/local/bin/heroku
"""

## Launch app
- heroku login
- heroku create <app-name>
- git push heroku master
    - git push heroku main
- heroku create -a example-app [auto adds heroku remote]
- git remote -v (checks ]
- app_exists: 
    - heroku git:remote -a example-app
- git push heroku main

## ide-yeet 
- Other 


<script src="https://gist.github.com/xmax1/f9f66535467ec44759193a18594e72c4.js"></script>