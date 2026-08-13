import SwiftUI
import llama

struct ContentView: View {
    // Test that we can initialize a llama context params struct
    let params = llama_context_default_params()

    var body: some View {
        VStack(spacing: 40) {
            Text("Llama Framework Test on tvOS")
                .font(.largeTitle)
                .padding()

            Text("llama_context_default_params() created successfully")
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding()

            // Display some param values to confirm the framework is working
            Text("n_ctx: \(params.n_ctx)")
                .font(.title2)

            Text("n_batch: \(params.n_batch)")
                .font(.title2)

            Spacer()
        }
        .padding(50)
        // Larger size suitable for TV display
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
