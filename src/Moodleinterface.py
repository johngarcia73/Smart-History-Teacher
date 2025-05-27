import requests
class MoodleInterfaceAgent:
   
    # Configurar autenticación
    MOODLE_URL = "https://tuits.moodle.com"
    MOODLE_TOKEN = "tu_token_secreto"
    USER_ID = 123

    def post_answer_to_forum(self, answer: str, course_id: int):
        requests.post(
            f"{MOODLE_URL}/webservice/rest/server.php",
            params={
                "wstoken": MOODLE_TOKEN,
                "wsfunction": "mod_forum_add_discussion_post",
                "post[subject]": "Respuesta ITS",
                "post[message]": answer,
                "courseid": course_id
            }
        )