La carpeta RAG_server conté tot el necessari per a posar en marxa un servidor que fa RAG sobre el US code. 
Tan sols cal clonar la carpeta, donar els permisos necessaris a setup.sh, i executar el fitxer. 
Automàticament crearà un venv, instal·larà tot el necessari i posarà en marxa el servidor. 

Per a utilitzar el servidor, cal fer una petició POST a l'endpoint /query amb el següent JSON:
{
    "prompt": "test",
    "key": "set_here"
}
El valor de "key" és la clau d'accés al servidor, que per ara és "set_here".
El servidor respondrà amb un JSON que conté el resultat de la consulta: 
{
    "result": "El resultat de la consulta"
    "metadata": "La metadada de la consulta"
}

Per provar des d'una consola de windows: 
curl http://IP:80/query -H "Content-Type: application/json" -d "{\"prompt\": \"test\", \"key\": \"set_here\"}"

A embed_utils hi ha les eines necessaries per a crear les dades que s'han ja emmagatzemat a store (els vectors i metadates necessàries). 
Es carreguen automàticament quan s'engega el servidor. Cal tenir en un .zip el USC. Es podria fer servir per altres fitxers, però caldria canviar el codi. 

Els fitxers estan en trossos macos suficientment petits com per a que github no es queixi. 