import React from 'react';
import { Text, StyleSheet, SafeAreaView, StatusBar, Button, View} from "react-native"
import { NavigationContainer } from '@react-nav'

export default WelcomeScreen()

const handlePress = () => console.log("Text pressed...")

function WelcomeScreen(props) {
    
    return (
        <SafeAreaView style={styles.container}>

            <View styles={styles.standardButton} >
                <Button title="Evaluate Your Cough" accessibilityLabel="Determine if your cough is a dry or wet cough" onPress="{handlePress}" />
            </View>
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
    standardButton: {
        width: '100%',
        height: 70,
    }
});
