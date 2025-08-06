# MODELO DE MATRIZ DE SIMILARIDADE POR EMBEDDINGS

Pense nos embeddings como uma espécie de **mapa estelar da linguagem**. Em vez de estrelas, temos palavras, e em vez de um espaço 3D, temos um espaço com centenas ou milhares de dimensões.

O modelo de linguagem não vê a palavra "professor" como um conjunto de letras. Ele a vê como um **vetor**: uma lista de números que representa um ponto único nesse mapa multidimensional. Ex: ``[0.23, -0.54, 0.81, ..., 0.12]``.

## Como essa localização é definida?

O princípio fundamental é a Hipótese Distribucional: palavras que aparecem em contextos semelhantes possuem significados semelhantes.

Durante o treinamento em trilhões de frases, o modelo analisa as palavras que cercam cada termo.

Em um exmeplo, onde as palavras são: *"professor"*, *"educador"*, *"estudante"*, *"ciências"*

- Ele nota que as palavras *"professor"* e *"educador"* aparecem frequentemente nos mesmos *"bairros"* de palavras, como *"escola"*, *"alunos"*, *"ensinar"*, *"universidade"*, *"sala de aula"* e *"pedagogia"*.
- Ele nota que *"estudante"* também aparece em muitos desses bairros, mas frequentemente em uma relação diferente (recebendo a ação, em vez de executá-la).
- Ele nota que *"ciências"* aparece em alguns desses contextos (*"professor de ciências"*, *"estudante de ciências"*), mas também em muitos outros, com palavras como *"laboratório",* *"experimento",* *"física",* *"química",* etc.

O modelo, então, ajusta os vetores (as coordenadas no mapa) de modo que as palavras que compartilham contextos fiquem **geometricamente próximas** umas das outras. As palavras *"professor"* e *"educador"* acabam sendo posicionadas quase no mesmo lugar no mapa. "Estudante" fica por perto, mas em uma posição distinta. "Ciências" fica em uma região próxima, mas mais afastada, na direção do *"conhecimento"* e *"assuntos acadêmicos"*.

## Medindo a Proximidade: Similaridade de Cossenos

Para medir o quão "próximas" duas palavras estão nesse mapa, não usamos a distância comum. Usamos a Similaridade de Cossenos, que mede o ângulo entre os vetores.

- **Um ângulo de 0° (cosseno = 1.0)** significa que os vetores apontam na mesma direção (significados quase idênticos).
- **Um ângulo de 90° (cosseno = 0.0)** significa que são ortogonais (sem relação de significado).

---

# Modelo de Matriz de Similaridade Semântica

Este projeto contém um script em Python que demonstra como calcular e visualizar a similaridade semântica entre um conjunto de palavras fornecidas pelo usuário. Utilizando a arquitetura de modelos Transformer, o script transforma palavras em **vetores de alta dimensão (embeddings)** e, em seguida, calcula a proximidade entre elas usando a similaridade de cossenos.

## Visão Geral

No campo do Processamento de Linguagem Natural (PNL), entender o significado e o contexto das palavras é um desafio fundamental. Este modelo aborda esse desafio através de duas etapas principais:
1. **Vetorização por Embeddings:** Cada palavra é convertida em um vetor numérico de alta dimensão. Este processo, conhecido como geração de embeddings, posiciona palavras com significados semelhantes em pontos próximos dentro de um "espaço de significados" vetorial. Palavras como "carro" e "automóvel", por exemplo, terão vetores muito próximos.
2. **Cálculo da Matriz de Similaridade:** Uma vez que cada palavra é um vetor, podemos medir a "distância" ou "proximidade" entre elas. Este script utiliza a Similaridade de Cossenos, uma métrica que calcula o cosseno do ângulo entre dois vetores.
    - Um score de ``1.0`` significa que as palavras são semanticamente idênticas ou muito próximas.
    - Um score de ``0.0`` indica ausência de similaridade.
O resultado final é uma matriz que exibe a pontuação de similaridade entre cada par de palavras na lista de entrada.

## Funcionamento

O script segue uma sequência lógica de operações:
1. **Coleta de Palavras:** O script é interativo e solicita que o usuário insira as palavras que deseja analisar, uma por vez.
2. **Carregamento do Modelo:** Carrega o modelo ``paraphrase-multilingual-MiniLM-L12-v2`` da biblioteca ``sentence-transformers``. Este é um modelo Transformer leve e eficiente, pré-treinado em mais de 50 idiomas, incluindo o português.
3. **Geração dos Embeddings:** O modelo processa a lista de palavras e gera um vetor numérico (embedding) para cada uma delas.
4. **Cálculo da Similaridade:** Utiliza a função ``cosine_similarity`` da biblioteca ``scikit-learn`` para calcular a similaridade entre todos os pares de vetores gerados.
5. **Visualização:** Usa a biblioteca ``pandas`` para criar e exibir uma tabela (DataFrame) clara e legível da matriz de similaridade resultante.

## Aplicações Práticas

A capacidade de quantificar a similaridade semântica tem inúmeras aplicações no mundo real, tais como:

- **Sistemas de Recomendação:** Recomendar produtos, artigos ou vídeos com base na similaridade de suas descrições.
- **Motores de Busca Avançados:** Encontrar documentos que são conceitualmente relacionados a uma busca, mesmo que não contenham as palavras-chave exatas.
- **Análise de Sentimento e Feedback:** Agrupar comentários de clientes com temas semelhantes para identificar pontos fortes e fracos de um produto.
- **Detecção de Plágio e Conteúdo Duplicado:** Identificar textos que são semanticamente muito parecidos.
- **Chatbots e Assistentes Virtuais:** Compreender a intenção do usuário, mesmo que ele use sinônimos ou formulações diferentes para a mesma pergunta.


## Tecnologias e Bibliotecas Utilizadas
- **Python 3.x**
- **Sentence-Transformers:** Para carregar o modelo Transformer e gerar os embeddings.
- **Scikit-learn:** Para o cálculo da similaridade de cossenos.
- **Pandas:** Para a formatação e exibição da matriz de resultado

## Como Usar

1. **Pré-requisitos:** Certifique-se de ter o ``Python 3`` instalado em seu sistema, ou acesso ao ``Google Colab`` (No último caso, podendo utiizar o arquivo ``.ipynb``).
   
2. **Instalação das Dependências:** Abra seu terminal ou prompt de comando e instale as bibliotecas necessárias com o pip:

```Bash
pip install sentence-transformers scikit-learn pandas
```

3. Salve o código em um arquivo (ex: analise_similaridade.py), ou utilize o arquivo ``modelo_de_embeddings_e_matriz_de_similaridade.py`` obtido por esse repositório e execute pelo terminal, dentro da pasta onde o arquivo está:

```Bash
python analise_similaridade.py
```

4. **Interação:** O script solicitará que você digite as palavras uma a uma. Após digitar cada palavra, pressione Enter. Quando terminar de adicionar palavras, simplesmente pressione ``Enter`` com o campo vazio para iniciar a análise.

## Exemplo de Saída

```Python

### MODELO DE MATRIZ DE SIMILARIDADE INICIADA ###

Recebendo palavras para modelo de vetorização...

Digite a palavra ou aperte enter terminar: professor
Digite a palavra ou aperte enter terminar: educador
Digite a palavra ou aperte enter terminar: estudante
Digite a palavra ou aperte enter terminar: ciências
Digite a palavra ou aperte enter terminar: 

Carregando o modelo de transformer...

Gerando os embeddings...

Calculando a matriz de similaridade...

Criando dataframe para visualização da matriz de similaridade com pandas...

Tabela de Similaridade Semântica (Similaridade de Cossenos):
           professor  educador  estudante  ciências       
professor      1.000     0.709      0.661     0.601  0.503
educador       0.709     1.000      0.755     0.550  0.535
estudante      0.661     0.755      1.000     0.565  0.523
ciências       0.601     0.550      0.565     1.000  0.605
               0.503     0.535      0.523     0.605  1.000
```

## Análise da Tabela

- **``professor`` e ``educador`` (0.709):** Como esperado, a similaridade é altíssima. As palavras são quase sinônimos em muitos contextos, então seus vetores no mapa da linguagem estão muito próximos.

- **``professor`` e ``estudante`` (0.661):** A similaridade é alta, mas visivelmente menor que a anterior. Eles pertencem ao mesmo domínio (educação), mas ocupam papéis diferentes e complementares. Estão no mesmo "bairro" do mapa, mas não na mesma casa.

- **``professor`` e ``ciências`` (0.601):** A conexão aqui também é forte. A combinação "professor de ciências" e "professor da ciência" ou campos da "ciência" para áreas de ensino é muito comum. O modelo entende que "ciências" é um campo de ensino para "professor".

- **``estudante`` e ``ciências`` (0.565):** A similaridade é moderada. Um "estudante" pode ou não estar estudando "ciências". A conexão existe, mas é menos intrínseca do que a relação entre "professor" e "ciências" (todo professor ensina algo, ou algum tipo de "ciência") ou "professor" e "estudante".


## Licença

Este projeto está sob a licença *Apache 2.0*. Veja o arquivo `LICENSE` para mais detalhes.

## Autor

Patrícia Canossa Gagliardi







