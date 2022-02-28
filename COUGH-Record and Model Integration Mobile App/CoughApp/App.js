import * as React from 'react';
import { SafeAreaView, StyleSheet, View, Alert, Text, Button} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Audio } from 'expo-av'
import { FontAwesome } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';
import { Stopwatch } from 'react-native-stopwatch-timer';
import * as FileSystem from "expo-file-system";
import Spinner from 'react-native-loading-spinner-overlay';

const Stack = createNativeStackNavigator();


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  boldText: {
    fontWeight: 'bold',
    textAlign: 'center',
  },
  spinnerTextStyle: {
    color: 'black',
  },
  instructionText: {
    fontWeight: 'normal',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  fill: {
    flex: 1,
    margin: 16
  },
  button: {
    margin: 16
  }
});


export default function App(){
  const [recording, setRecording] = React.useState();
  const [recordings, setRecordings] = React.useState([]);
  const [message, setMessage] = React.useState("");
  const [loading, setLoading] = React.useState(false);

  // stop watch
  const [isStopwatchStart, setIsStopwatchStart] = React.useState(false);
  const [resetStopwatch, setResetStopwatch] = React.useState(false);

  // API to Run Model
  const MODEL_BACKEND = "https://covidcough.ue.r.appspot.com/evaluatecough/";

  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="RecordCough" component={RecordCoughScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
  

  function HomeScreen({navigation}) {
    return (
      <View style={styles.container} >
        <Text>Welcome to the COVID Cough Application</Text>
        <Button title="Record Cough" onPress = {() => navigation.navigate('RecordCough')} />
      </View>
    );
  }
  
  function RecordCoughScreen({navigation}) {
    return (
      <SafeAreaView style={{ flex: 1 }}>
        <View style={styles.container}>
          <Spinner visible={loading} textContent={'Inspecting...'} textStyle={styles.spinnnerTextStyle} />
          <Text>{message}</Text>
          <Text style={styles.boldText}>Instructions</Text>
          <Text></Text>
          <Text style={styles.instructionText}>When ready click the button 'Start Recording' to record your cough. The cough will be recorded for 10 seconds. Please try and maintain the cough for the full 10 seconds and find a quiet spot with minimal background noise.</Text>
          <Text></Text>
          <FontAwesome name="microphone" size={36} color='black' />
          <Button title={recording ? 'Stop Recording' : 'Start Recording'} onPress={recording ? handleRecordCoughStop : handleRecordCoughStart} />
          <Stopwatch laps msecs start={isStopwatchStart}
            reset={resetStopwatch}
            options={options}
            getTime={(time) => { console.log(time); }} 
          />
          <Text></Text>
          <Text></Text>
          <Text>Recordings</Text>
          {getRecordingLines()}
          <StatusBar style="auto" />
        </View>
      </SafeAreaView>
    );
  }
  
  async function handleRecordCoughStart(){
    // console.log('Starting to record cough');
    try{
      console.log('Requesting permissions..');
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      console.log('Starting recording..');
      setResetStopwatch(true);
      setIsStopwatchStart(true);
      const { recording } = await Audio.Recording.createAsync({
        ios: {
          extension: '.wav',
          sampleRate: 22050,
          numberOfChannels: 1,
          audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH,
          outputFormat: Audio.RECORDING_OPTION_IOS_OUTPUT_FORMAT_LINEARPCM
        },
        android: {
          extension: '.wav',
          sampleRate: 22050,
          numberOfChannels: 1,
          audioQuality: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_AAC,
          outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_DEFAULT
        }
      }
      );
      setRecording(recording);
      console.log('Recording started')
    } catch(err) {
      setMessage('Failed to start recording', err);
      console.error('Failed to start recording', err);
    }
  }
  
  async function handleRecordCoughStop() {
    console.log('Stopping recording..')
    setIsStopwatchStart(false);
    //setResetStopwatch(false);
    setRecording(undefined);
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    console.log('Recording stopped: ', uri)
    let updatedRecordings = [];
    const { sound, status } = await recording.createNewLoadedSoundAsync();
    updatedRecordings.push({
      sound: sound,
      duration: getDurationFormatted(status.durationMillis),
      file: recording.getURI()
    })

    setRecordings(updatedRecordings);
  }

  function getDurationFormatted(millis) {
    const minutes = millis / 1000 / 60;
    const minutesDisplay = Math.floor(minutes);
    const seconds = Math.round((minutes - minutesDisplay) * 60);
    const secondsDisplay = seconds < 10 ? `0${seconds}` : seconds;
    return `${minutesDisplay}:${secondsDisplay}`;
  }

  async function evaluatecough(audio_recording){
    console.log('Starting API call');
    console.log('Audio File:  ' + audio_recording.file);
    setLoading(true);
    try{
      const response = await FileSystem.uploadAsync(MODEL_BACKEND, audio_recording.file);
      const body = JSON.parse(response.body);
      console.log(body);
      const response_info = () => Alert.alert('Results', 'Cough Type: ' + body.class + '\nProbability Dry: ' + body.percent_dry + '%\nProbability Wet: ' + body.percent_wet + '%', [
        {
          text: 'OK', onPress: () => console.log('OK Pressed')
        }
      ]);

      response_info();

    } catch(err) {
      const error_info = () => Alert.alert('Error', 'There was an error evaluating your cough. Please contact support!\n\n' + err, [
        {
          text: 'OK', onPress: () => console.log('OK Pressed')
        }
      ]);

      error_info();

      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function playsound(recordingLine){
    try{
      recordingLine.sound.replayAsync();
    } catch(err) {
      const error_info = () => Alert.alert('Error', 'There was an error playing your cough. Please contact support!\n\n' + err, [
        {
          text: 'OK', onPress: () => console.log('OK Pressed')
        }
      ]);

      error_info();
    }
    
  }

  function getRecordingLines() {
    return recordings.map((recordingLine, index) => {
      return (
        <View key={index} style={styles.row}>
          <Text style={styles.fill}>{recordingLine.duration}</Text>
          <Button style={styles.button} onPress={() => playsound(recordingLine)} title="Play"></Button>
          <Button style={styles.button} onPress={() => evaluatecough(recordingLine)} title="Evaluate"></Button>
        </View>
      );
    });
  }
}

const options = {
  container: {
    backgroundColor: 'red',
    padding: 5,
    borderRadius: 5,
    width: 200,
    alignItems: 'center',
  },
  text: {
    fontSize: 25,
    color: 'white',
    marginLeft: 7,
  },
};