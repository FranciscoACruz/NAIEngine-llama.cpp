import SwiftUI
import llama

struct ContentView: View {
    // Test that we can initialize a llama context params struct
    let params = llama_context_default_params()

    var body: some View {
        VStack(spacing: 20) {
            Text("Llama Framework Test")
                .font(.largeTitle)
                .padding()

            Text("llama_context_default_params() created successfully")
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding()

            // Display some param values to confirm the framework is working
            Text("n_ctx: \(params.n_ctx)")
                .font(.body)

            Text("n_batch: \(params.n_batch)")
                .font(.body)

            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
