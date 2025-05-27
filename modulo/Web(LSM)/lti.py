from flask import Flask, request,session,render_template_string
from lti import ToolProvider,OutcomeRequest

import logging

import os

app = Flask(__name__)
app.secret_key= 'Clave_Secreta'
logging.basicConfig(level=logging.DEBUG)

@app.route('/lti', methods=['POST'])
def lti_launch():
    lti_params = {
        'consumer_key': os.environ.get('lti_Consumer_key'),
        'consumer_secret': os.environ.get('lti_Consumer_secret')
    }
    tool_provider= ToolProvider.from_flask_request(
        secret= lti_params['consumer_secret'],
        request=request
    )
    if not tool_provider.is_valid_request(request):
        return 'Firma OAuth invalida', 401

    session['user_id']= request.form.get('user_id')
    session['course_id']= request.form.get('context_id')

    return render_template_string('''
        <h1>¡Bienvenido al Tutor Inteligente!</h1>
        <p>Usuario: {{ user_id }}</p>
        <p>Curso: {{ course_id }}</p>
    ''', user_id=session['user_id'], course_id=session['course_id'])



@app.route('/submit-grade', methods=['POST'])
def submit_grade():
    # Obtener datos de la sesión
    user_id = session.get('user_id')
    course_id = session.get('course_id')

    # Crear objeto OutcomeRequest
    outcome = OutcomeRequest({
        'consumer_key': 'tu_consumer_key',
        'consumer_secret': 'tu_consumer_secret',
        'lis_outcome_service_url': request.form.get('lis_outcome_service_url'),
        'lis_result_sourcedid': request.form.get('lis_result_sourcedid')
    })

    # Enviar calificación (ej: 0.8 = 80%)
    response = outcome.post_replace_result(0.8)

    if response.success:
        return 'Calificación enviada exitosamente'
    else:
        return 'Error al enviar calificación', 500


if __name__ == '__main__':
    app.run(ssl_context='adhoc', debug=True)