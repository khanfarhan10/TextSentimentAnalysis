import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class Result {

  String data;

  Result({this.data});

  factory Result.fromJson(Map<String, dynamic> json) {
    return Result(
      data: json['data'],
    );
  }
}



class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(title: 'Text Sentiment Analysis'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  String task = "Sentiment Analysis";
  TextEditingController textController = new TextEditingController();
  Result fetch_data;

  Future<Result> futureAlbum;

  @override
  void initState() {
    super.initState();
    futureAlbum = postSentence();
  }

  Widget _returnButton(String t) {

    if(t == "Sentiment Analysis") {
      return Text('Analyze', style: TextStyle(fontSize: 20));
    }
    else if(t == "Summarization") {
      return Text('Summarize', style: TextStyle(fontSize: 20));
    }
    else {
      return Text('Paraphrase', style: TextStyle(fontSize: 20));
    }

  }

  Widget showText() {

    if(fetch_data == null) {
      return Text("");
    }
    else {
      return Text(fetch_data.data);
    }

  }

  Future<Result> postSentence() async {

    String appendText = "";

    if(task == "Sentiment Analysis") {
      appendText += "run_forward";
    }
    else if(task == "Summarization") {
      appendText += "run_forward_summarizer";
    }
    else {
      appendText += "run_forward_paraphrase";
    }


    final http.Response response = await http.post(
      'http://192.168.0.100:5000/' + appendText,
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, String>{
        'sentence': textController.text,
      }),
    );

    return Result.fromJson(jsonDecode(response.body));
    if(json.decode(response.body) != null) {
      return Result.fromJson(jsonDecode(response.body));
    }
    else {
      throw Exception('Unable to connect to backend');
    }


  }


  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(

        title: Text(widget.title),
      ),

      drawer: Drawer(
        child: ListView(
          // Important: Remove any padding from the ListView.
          padding: EdgeInsets.zero,
          children: <Widget>[
            DrawerHeader(
              child: Text('Utilities',style: TextStyle(fontSize: 30),),
              decoration: BoxDecoration(
                color: Colors.blue,
              ),
            ),
            ListTile(
              title: Text('Sentiment Analysis'),
              onTap: () {
                // Update the state of the app.
                // ...
                setState(() {
                  task = "Sentiment Analysis";
                });

                Navigator.pop(context);

              },
            ),
            ListTile(
              title: Text('Summarization'),
              onTap: () {
                // Update the state of the app.
                // ...
                setState(() {
                  task = "Summarization";
                });
                Navigator.pop(context);
              },

            ),
            ListTile(
              title: Text('Paraphrase'),
              onTap: () {
                // Update the state of the app.
                // ...
                setState(() {
                  task = "Paraphrase";
                });
                Navigator.pop(context);
              },

            ),
          ],
        ),
      ),

      body: Center(
        child: Column(

          children: <Widget>[
            const SizedBox(height: 80),
              TextField(
                controller: textController,
                maxLines: 6,
                decoration: InputDecoration(
                  border: OutlineInputBorder(),
                  hintText: 'Enter your text',
              ),
            ),
            const SizedBox(height: 30),
            RaisedButton(
              onPressed: () {
                 postSentence().then((value) =>

                 setState(() {
                   fetch_data = value;
                 })

                 );
              },
              child: _returnButton(task),
            ),
            const SizedBox(height: 30),

            Container(
              child: showText()

            )

          ],
        ),
      ),
    );
  }
}
