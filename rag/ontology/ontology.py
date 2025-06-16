import os
import logging
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL
import requests
import re

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OntologyManager")

class OntologyManager:
    def __init__(self, ontology_file="historia_ontology.ttl"):
        self.ontology_file = ontology_file
        self.kg = Graph()
        self.HIST = Namespace("http://tutorhistoria.org/ontology#")
        self.ENT = Namespace("http://tutorhistoria.org/entities/")
        
        # Registrar namespaces
        self.kg.bind("hist", self.HIST)
        self.kg.bind("ent", self.ENT)
        
        # Verificar si la ontología ya existe
        if os.path.exists(self.ontology_file):
            self.load_ontology()
            logger.info("Ontología cargada desde archivo")
        else:
            self.create_base_ontology()
            #self.populate_initial_data()
            self.poblar_ontologia()
            self.save_ontology()
            logger.info("Ontología base creada y poblada")
    
    def create_base_ontology(self):
        """Crea la estructura básica de la ontología (sin datos)"""
        # Definir clases básicas
        clases = [
            self.HIST.EventoHistorico,
            self.HIST.Personaje,
            self.HIST.Ubicacion,
            self.HIST.Periodo,
            self.HIST.Movimiento,
            self.HIST.Concepto
        ]
        
        for clase in clases:
            self.kg.add((clase, RDF.type, OWL.Class))
        
        # Definir propiedades
        propiedades = [
            (self.HIST.tieneParticipante, self.HIST.EventoHistorico, self.HIST.Personaje),
            (self.HIST.tieneLugar, self.HIST.EventoHistorico, self.HIST.Ubicacion),
            (self.HIST.ocurreEn, self.HIST.EventoHistorico, self.HIST.Periodo),
            (self.HIST.influenciadoPor, self.HIST.EventoHistorico, self.HIST.EventoHistorico)
        ]
        
        for prop, dominio, rango in propiedades:
            self.kg.add((prop, RDF.type, OWL.ObjectProperty))
            self.kg.add((prop, RDFS.domain, dominio))
            self.kg.add((prop, RDFS.range, rango))
        
        # Propiedad adicional para Personaje
        self.kg.add((self.HIST.influencio, RDF.type, OWL.ObjectProperty))
        self.kg.add((self.HIST.influencio, RDFS.domain, self.HIST.Personaje))
        self.kg.add((self.HIST.influencio, RDFS.range, self.HIST.Personaje))
        
        # Definir periodos históricos
        periodos = [
            ("Prehistoria", -3500000, -3000),
            ("Edad_Antigua", -3000, 476),
            ("Edad_Media", 476, 1453),
            ("Edad_Moderna", 1453, 1789),
            ("Edad_Contemporanea", 1789, None)
        ]
        
        for nombre, inicio, fin in periodos:
            periodo_uri = self.HIST[nombre]
            self.kg.add((periodo_uri, RDF.type, OWL.Class))
            self.kg.add((periodo_uri, RDFS.subClassOf, self.HIST.Periodo))
            self.kg.add((periodo_uri, self.HIST.inicioPeriodo, Literal(inicio)))
            if fin:
                self.kg.add((periodo_uri, self.HIST.finPeriodo, Literal(fin)))
    
    def load_ontology(self):
        """Carga la ontología desde archivo"""
        self.kg.parse(self.ontology_file, format="turtle")
    
    def save_ontology(self):
        """Guarda la ontología en archivo"""
        self.kg.serialize(self.ontology_file, format="turtle")
    
    def add_historical_event(self, nombre, fecha_inicio=None, participantes=None, lugares=None):
        """Añade un evento histórico a la ontología"""
        evento_uri = URIRef(f"{self.ENT}{nombre.replace(' ', '_')}")
        
        # Verificar si ya existe
        if (evento_uri, None, None) in self.kg:
            logger.warning(f"Evento '{nombre}' ya existe en la ontología")
            return evento_uri
        
        self.kg.add((evento_uri, RDF.type, self.HIST.EventoHistorico))
        self.kg.add((evento_uri, RDFS.label, Literal(nombre)))
        
        if fecha_inicio:
            self.kg.add((evento_uri, self.HIST.fechaInicio, Literal(fecha_inicio)))
        
        if participantes:
            for persona in participantes:
                persona_uri = URIRef(f"{self.ENT}{persona.replace(' ', '_')}")
                self.kg.add((persona_uri, RDF.type, self.HIST.Personaje))
                self.kg.add((persona_uri, RDFS.label, Literal(persona)))
                self.kg.add((evento_uri, self.HIST.tieneParticipante, persona_uri))
        
        if lugares:
            for lugar in lugares:
                lugar_uri = URIRef(f"{self.ENT}{lugar.replace(' ', '_')}")
                self.kg.add((lugar_uri, RDF.type, self.HIST.Ubicacion))
                self.kg.add((lugar_uri, RDFS.label, Literal(lugar)))
                self.kg.add((evento_uri, self.HIST.tieneLugar, lugar_uri))
        
        logger.info(f"Evento '{nombre}' añadido a la ontología")
        return evento_uri
    
    def relate_events(self, evento1, evento2, relacion):
        """Establece una relación entre dos eventos históricos"""
        uri1 = URIRef(f"{self.ENT}{evento1.replace(' ', '_')}")
        uri2 = URIRef(f"{self.ENT}{evento2.replace(' ', '_')}")
        
        # Verificar si ambos eventos existen
        if not ((uri1, None, None) in self.kg and (uri2, None, None) in self.kg):
            logger.error("Uno o ambos eventos no existen en la ontología")
            return False
        
        self.kg.add((uri1, self.HIST[relacion], uri2))
        logger.info(f"Relación '{relacion}' establecida entre '{evento1}' y '{evento2}'")
        return True
    
    def infer_historical_period(self, evento_uri):
        """Infere el periodo histórico de un evento basado en su fecha"""
        fecha = self.kg.value(evento_uri, self.HIST.fechaInicio)
        if not fecha:
            return None
        
        try:
            año = int(str(fecha).split('-')[0])
        except ValueError:
            return None
        
        if año < -3000:
            return self.HIST.Prehistoria
        elif -3000 <= año < 476:
            return self.HIST.Edad_Antigua
        elif 476 <= año < 1453:
            return self.HIST.Edad_Media
        elif 1453 <= año < 1789:
            return self.HIST.Edad_Moderna
        else:
            return self.HIST.Edad_Contemporanea
    
    import re

    def expand_query(self, query, max_terms=3):
        """
        Expande la consulta reemplazando el término del evento histórico encontrado
        por el mismo término seguido de una indicación entre paréntesis con información
        derivada de las relaciones definidas (por ejemplo, influenciadoPor, causaDirecta, 
        parteDe y resultadoEn).

        Ejemplo:
        Entrada: "Cómo fue la Revolución Francesa en Europa?"
        Salida: "Cómo fue la Revolución Francesa (si cabe, tener en cuenta la influencia en Revolución Americana) en Europa?"
        """
        if not query:
            return query

        expanded_query = query  # Copia de la query original
        rel_types = [
            self.HIST.influenciadoPor,
            self.HIST.causaDirecta,
            self.HIST.parteDe,
            self.HIST.resultadoEn
        ]

        # Recorremos todos los eventos históricos de la ontología
        for event in self.kg.subjects(RDF.type, self.HIST.EventoHistorico):
            label = self.kg.value(event, RDFS.label)
            if not label:
                continue

            label_str = str(label)
            # Si el término del evento aparece (sin distinguir mayúsculas/minúsculas) en la query...
            if re.search(re.escape(label_str), query, flags=re.IGNORECASE):
                expansion_phrases = []
                term_count = 0

                # Para cada tipo de relación definido, buscamos hasta max_terms relaciones
                for rel in rel_types:
                    if term_count >= max_terms:
                        break

                    # Por cada relación del evento:
                    for _, _, related in self.kg.triples((event, rel, None)):
                        related_label = self.kg.value(related, RDFS.label)
                        if not related_label:
                            continue
                        # Generar el texto específico según el tipo de relación
                        if rel == self.HIST.influenciadoPor:
                            phrase = f"si cabe, tener en cuenta la influencia en {related_label}"
                        elif rel == self.HIST.causaDirecta:
                            phrase = f"si cabe, tener en cuenta la causa directa sobre {related_label}"
                        elif rel == self.HIST.parteDe:
                            phrase = f"si cabe, tener en cuenta su relación con {related_label}"
                        elif rel == self.HIST.resultadoEn:
                            phrase = f"si cabe, tener en cuenta que tuvo resultado en {related_label}"
                        else:
                            phrase = ""
                        
                        if phrase and phrase not in expansion_phrases:
                            expansion_phrases.append(phrase)
                            term_count += 1
                            # Pasamos a la siguiente relación; si se desea capturar más de una por tipo, quitar 'break'
                            break
                if expansion_phrases:
                    # Formar la cadena de expansión
                    expansion_text = " (" + ", ".join(expansion_phrases) + ")"
                    # Reemplazar el primer match de label en la query con label + expansion_text
                    expanded_query = re.sub(re.escape(label_str), f"{label_str}{expansion_text}", expanded_query, flags=re.IGNORECASE)
                    
        print(f"La consulta mejorada es: {expanded_query}")
        return expanded_query

        
    def get_related_events(self, event_name, relation_type):
        """Obtiene eventos relacionados de un tipo específico"""
        event_uri = URIRef(f"{self.ENT}{event_name.replace(' ', '_')}")
        related_events = []
        
        for s, p, o in self.kg.triples((event_uri, self.HIST[relation_type], None)):
            label = str(self.kg.value(o, RDFS.label))
            related_events.append(label)
        
        return related_events

    def poblar_ontologia(self):
        """Método para poblar la ontología con datos iniciales (ejemplo)"""
        # Aquí puedes agregar datos iniciales a la ontología
        self.add_historical_event("Caida del Imperio Romano", fecha_inicio="476", participantes=["Odoacro"], lugares=["Roma"])
        self.add_historical_event("Descubrimiento de América", fecha_inicio="1492", participantes=["Cristóbal Colón"], lugares=["San Salvador"])

   

    # Poblar ontología
    def poblar_ontologia(self):
        """Pobla la ontología con datos históricos completos de forma manual"""
        # ======================
        # PERIODOS HISTÓRICOS
        # ======================
        periodos = [
            ("Prehistoria", -3500000, -3000),
            ("Edad_Antigua", -3000, 476),
            ("Edad_Media", 476, 1453),
            ("Edad_Moderna", 1453, 1789),
            ("Edad_Contemporanea", 1789, None)
        ]
        
        for nombre, inicio, fin in periodos:
            periodo_uri = self.HIST[nombre]
            self.kg.add((periodo_uri, RDF.type, self.HIST.Periodo))
            self.kg.add((periodo_uri, RDFS.label, Literal(nombre.replace('_', ' '))))
            self.kg.add((periodo_uri, self.HIST.inicioPeriodo, Literal(inicio)))
            if fin:
                self.kg.add((periodo_uri, self.HIST.finPeriodo, Literal(fin)))
        
        # ======================
        # MOVIMIENTOS HISTÓRICOS
        # ======================
        movimientos = [
            ("Revolución_Neolítica", -10000, -3000, "Transición a agricultura"),
            ("Civilización_Egipcia", -3100, -332, "Antiguo Egipto"),
            ("Civilización_Griega", -800, -146, "Antigua Grecia"),
            ("Imperio_Romano", -27, 476, "Roma antigua"),
            ("Renacimiento", 1300, 1600, "Renacimiento cultural"),
            ("Ilustración", 1685, 1815, "Movimiento intelectual"),
            ("Revolución_Industrial", 1760, 1840, "Industrialización")
        ]
        
        for nombre, inicio, fin, desc in movimientos:
            movimiento_uri = self.HIST[nombre]
            self.kg.add((movimiento_uri, RDF.type, self.HIST.Movimiento))
            self.kg.add((movimiento_uri, RDFS.label, Literal(nombre.replace('_', ' '))))
            self.kg.add((movimiento_uri, self.HIST.descripcion, Literal(desc)))
            self.kg.add((movimiento_uri, self.HIST.inicioPeriodo, Literal(inicio)))
            self.kg.add((movimiento_uri, self.HIST.finPeriodo, Literal(fin)))
        
        # ======================
        # EVENTOS HISTÓRICOS CLAVE
        # ======================
        eventos = [
            # Prehistoria y Edad Antigua
            ("Revolución_Neolítica", "-10000", [], ["Creciente_Fértil"]),
            ("Construcción_Pirámides_Giza", "-2560", ["Faraón_Keops"], ["Guiza", "Egipto"]),
            ("Fundación_de_Roma", "-753", ["Rómulo", "Remo"], ["Roma"]),
            ("Guerras_Púnicas", "-264", ["Aníbal", "Escipión_Africano"], ["Cartago", "Roma"]),
            
            # Edad Media
            ("Caída_Imperio_Romano_Oeste", "476", ["Rómulo_Augústulo"], ["Roma"]),
            ("Primera_Cruzada", "1096", ["Godofredo_de_Bouillón"], ["Jerusalén"]),
            ("Peste_Negra", "1347", [], ["Europa"]),
            
            # Edad Moderna
            ("Caída_Constantinopla", "1453", ["Constantino_XI"], ["Constantinopla"]),
            ("Descubrimiento_de_América", "1492", ["Cristóbal_Colón"], ["Guanahaní", "América"]),
            ("Reforma_Protestante", "1517", ["Martín_Lutero"], ["Wittenberg"]),
            ("Revolución_Científica", "1543", ["Galileo_Galilei", "Isaac_Newton"], ["Europa"]),
            
            # Edad Contemporánea
            ("Revolución_Industrial", "1760", ["James_Watt"], ["Inglaterra"]),
            ("Revolución_Americana", "1775", ["George_Washington", "Thomas_Jefferson"], ["Boston", "Filadelfia"]),
            ("Revolución_Francesa", "1789", ["Luis_XVI", "Robespierre", "Napoleón_Bonaparte"], ["París", "Versalles"]),
            ("Primera_Guerra_Mundial", "1914", ["Guillermo_II", "Woodrow_Wilson"], ["Europa"]),
            ("Segunda_Guerra_Mundial", "1939", ["Adolf_Hitler", "Winston_Churchill"], ["Europa", "Pacífico"]),
            ("Caída_Muro_Berlín", "1989", ["Mijaíl_Gorbachov"], ["Berlín"])
        ]
        
        for nombre, fecha, participantes, lugares in eventos:
            evento_uri = self.add_historical_event(
                nombre.replace('_', ' '),
                fecha_inicio=fecha,
                participantes=participantes,
                lugares=lugares
            )
        
        # ======================
        # PERSONAJES HISTÓRICOS
        # ======================
        personajes = [
            ("Alejandro_Magno", "-356", "-323", "Rey macedonio"),
            ("Julio_César", "-100", "-44", "Dictador romano"),
            ("Cleopatra", "-69", "-30", "Última faraona"),
            ("Carlomagno", "742", "814", "Emperador del Sacro Imperio"),
            ("Leonardo_da_Vinci", "1452", "1519", "Artista renacentista"),
            ("Napoleón_Bonaparte", "1769", "1821", "Emperador francés"),
            ("Simón_Bolívar", "1783", "1830", "Libertador americano"),
            ("Abraham_Lincoln", "1809", "1865", "Presidente estadounidense"),
            ("Marie_Curie", "1867", "1934", "Científica pionera")
        ]
        
        for nombre, nac, muerte, desc in personajes:
            personaje_uri = URIRef(f"{self.ENT}{nombre}")
            self.kg.add((personaje_uri, RDF.type, self.HIST.Personaje))
            self.kg.add((personaje_uri, RDFS.label, Literal(nombre.replace('_', ' '))))
            self.kg.add((personaje_uri, self.HIST.descripcion, Literal(desc)))
            self.kg.add((personaje_uri, self.HIST.fechaNacimiento, Literal(nac)))
            self.kg.add((personaje_uri, self.HIST.fechaFallecimiento, Literal(muerte)))
        
        # ======================
        # UBICACIONES HISTÓRICAS
        # ======================
        ubicaciones = [
            ("Mesopotamia", "Región histórica", "Asia"),
            ("Valle_del_Indo", "Civilización antigua", "Asia"),
            ("Atenas", "Ciudad griega", "Europa"),
            ("Roma", "Capital imperial", "Europa"),
            ("Tenochtitlán", "Capital azteca", "América"),
            ("Machu_Picchu", "Ciudad inca", "América")
        ]
        
        for nombre, tipo, continente in ubicaciones:
            ubicacion_uri = URIRef(f"{self.ENT}{nombre}")
            self.kg.add((ubicacion_uri, RDF.type, self.HIST.Ubicacion))
            self.kg.add((ubicacion_uri, RDFS.label, Literal(nombre.replace('_', ' '))))
            self.kg.add((ubicacion_uri, self.HIST.tipoUbicacion, Literal(tipo)))
            self.kg.add((ubicacion_uri, self.HIST.continente, Literal(continente)))
        
        # ======================
        # CONCEPTOS HISTÓRICOS
        # ======================
        conceptos = [
            ("Democracia", "Sistema de gobierno del pueblo"),
            ("Monarquía", "Gobierno por reyes"),
            ("República", "Sistema representativo"),
            ("Colonialismo", "Dominación territorial"),
            ("Derechos_Humanos", "Derechos fundamentales")
        ]
        
        for nombre, desc in conceptos:
            concepto_uri = self.HIST[nombre]
            self.kg.add((concepto_uri, RDF.type, self.HIST.Concepto))
            self.kg.add((concepto_uri, RDFS.label, Literal(nombre.replace('_', ' '))))
            self.kg.add((concepto_uri, self.HIST.descripcion, Literal(desc)))
        
        # ======================
        # RELACIONES CLAVE
        # ======================
        relaciones = [
            # Relaciones entre eventos
            ("Revolución Francesa", "Revolución Americana", "influenciadoPor"),
            ("Primera Guerra Mundial", "Segunda Guerra Mundial", "causaDirecta"),
            ("Revolución Industrial", "Imperialismo", "contribuyoA"),
            
            # Personajes en eventos
            ("Napoleón Bonaparte", "Revolución Francesa", "participante"),
            ("George Washington", "Revolución Americana", "participante"),
            ("Adolf Hitler", "Segunda Guerra Mundial", "participante"),
            
            # Eventos en movimientos
            ("Reforma Protestante", "Renacimiento", "parteDe"),
            ("Caída Muro Berlín", "Guerra Fría", "parteDe"),
            
            # Relaciones entre personajes
            ("Julio César", "Cleopatra", "relacionadoCon"),
            ("Simón Bolívar", "Napoleón Bonaparte", "influenciadoPor"),
            
            # Conceptos relacionados
            ("Democracia", "Civilización Griega", "originadoEn"),
            ("Derechos Humanos", "Revolución Francesa", "consolidadoEn")
        ]
        
        for item1, item2, relacion in relaciones:
            try:
                if relacion == "participante":
                    persona_uri = URIRef(f"{self.ENT}{item1.replace(' ', '_')}")
                    evento_uri = URIRef(f"{self.ENT}{item2.replace(' ', '_')}")
                    self.kg.add((evento_uri, self.HIST.tieneParticipante, persona_uri))
                
                elif relacion == "parteDe":
                    evento_uri = URIRef(f"{self.ENT}{item1.replace(' ', '_')}")
                    movimiento_uri = self.HIST[item2.replace(' ', '_')]
                    self.kg.add((evento_uri, self.HIST.parteDe, movimiento_uri))
                
                else:
                    uri1 = URIRef(f"{self.ENT}{item1.replace(' ', '_')}")
                    uri2 = URIRef(f"{self.ENT}{item2.replace(' ', '_')}")
                    self.kg.add((uri1, self.HIST[relacion], uri2))
                    
            except Exception as e:
                logger.warning(f"Error añadiendo relación {relacion} entre {item1} y {item2}: {str(e)}")
        
        # ======================
        # ASIGNAR PERIODOS
        # ======================
        asignaciones_periodos = {
            "Prehistoria": ["Revolución Neolítica"],
            "Edad_Antigua": ["Construcción Pirámides Giza", "Fundación de Roma", "Guerras Púnicas"],
            "Edad_Media": ["Caída Imperio Romano Oeste", "Primera Cruzada", "Peste Negra"],
            "Edad_Moderna": ["Caída Constantinopla", "Descubrimiento de América", "Reforma Protestante"],
            "Edad_Contemporanea": ["Revolución Industrial", "Revolución Francesa", "Primera Guerra Mundial", "Segunda Guerra Mundial"]
        }
        
        for periodo, eventos in asignaciones_periodos.items():
            periodo_uri = self.HIST[periodo]
            for evento_nombre in eventos:
                evento_uri = URIRef(f"{self.ENT}{evento_nombre.replace(' ', '_')}")
                self.kg.add((evento_uri, self.HIST.ocurreEn, periodo_uri))
        
        

def descargar_datos_wikidata():
    query = """
    SELECT ?person ?personLabel ?birth ?death ?occupationLabel
    WHERE {
        ?person wdt:P31 wd:Q5;
                wdt:P106 ?occupation;
                wdt:P569 ?birth.
        OPTIONAL { ?person wdt:P570 ?death }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "es". }
    }
    LIMIT 1000
    """
    url = f"https://query.wikidata.org/sparql?query={query}&format=json"
    return requests.get(url).json()


if __name__ == "__main__":
    ontology = OntologyManager()
    query = "Cómo fue la Revolución Francesa, y quién la inició?"
    query = ontology.expand_query(query)
    print(f"La query expandida es: {query}")