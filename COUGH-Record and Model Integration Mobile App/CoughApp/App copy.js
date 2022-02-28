import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, SafeAreaView} from 'react-native';
import { createStack}
export default function App() {
  //const handlePress = () => console.log("Button pressed");
  function handlePress(){
    console.log("Button pressed take 1");
  }

  return (
    <SafeAreaView style={styles.container}>
       <Button title="Click to evaluate cough" onPress={handlePress} />
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
