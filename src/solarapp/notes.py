"""Model assumptions and limitations (shown in the app and in the report)."""

LIMITATIONS = [
    "Dados da NASA POWER são estimativas de satélite/reanálise (grade ~0,5° x 0,625°): "
    "não substituem medições de superfície, e a incerteza local pode ser de vários %.",
    "A irradiação mensal no plano inclinado usa o método de Klein: um 'dia médio' por mês "
    "(D&B Tab. 1.6.1) e a correlação difusa/global, o que suaviza a variabilidade diária.",
    "Modelos de transposição: isotrópico (Liu-Jordan) e Hay-Davies; ambos ignoram o "
    "espalhamento circunsolar/horizonte do modelo de Perez e sombreamento por obstáculos.",
    "Albedo constante (padrão 0,2), sem neve, sem variação sazonal.",
    "Correlações de fração difusa (Erbs, Orgill-Hollands, Collares-Pereira–Rabl) foram "
    "ajustadas a dados de outras regiões; Orgill-Hollands é horária e aqui aplicada ao Kt diário.",
    "'Horas de sol' é a duração astronômica do dia (sol acima do horizonte), não a insolação "
    "medida (heliógrafo), que é menor por causa de nuvens.",
    "Geração FV: P x HT x PR, com PR constante; não modela temperatura, sujeira ou inversor "
    "hora a hora. Coletor térmico: eficiência média constante (sem curva Hottel-Whillier-Bliss).",
    "Hora solar local (LST) da NASA POWER vale para a faixa de 15° de longitude, podendo diferir "
    "da hora legal; fuso e horário de verão não são tratados.",
    "Constante solar 1361 W/m² (os exemplos de Duffie & Beckman usam 1367 W/m²: "
    "diferença de 0,4%).",
    "Atlas INPE/LABREN: só importação de arquivo (não há API pública); valores do Atlas em "
    "Wh/m²/dia devem ser convertidos na importação (opção de unidade).",
]