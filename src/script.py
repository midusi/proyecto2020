from nn import StyleTransfer

STYLE_PATH = '../resources/imagen2.jpg'
CONTENT_PATH = '../resources/hielo.jpg'

st = StyleTransfer()

print('Cargando estilo ...')
btlnck = st.bottleneck(STYLE_PATH)
print('Estilo cargado.')

print('Cargando contenido ...')
image = st.load_img(CONTENT_PATH)
print('Contenido cargado.')

print('Preprocesando contenido ...')
content_image = st.preprocess_image(image, 384)
print('Contenido preprocesado.')

print('Aplicando estilo ...')
result = st.transform_style(btlnck, content_image)
print('Estilo aplicado.')

print(f'Resultado: {type(result)} \t Shape:{result.shape}')
