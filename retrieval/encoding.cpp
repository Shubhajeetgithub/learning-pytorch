#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <bitset>

// Encodes a standard unsigned 32-bit integer into ULEB128 (Base-128)
std::vector<uint8_t> encodeULEB128(uint32_t value) {
    std::vector<uint8_t> encoded;
    do {
        uint8_t byte = value & 0x7F; // Extract the lowest 7 bits (payload)
        value >>= 7;                // Shift right by 7 bits to process next chunk
        
        if (value != 0) {
            byte |= 0x80;            // Set Continuation Bit (Bit 7 = 1) if more bytes follow
        }
        encoded.push_back(byte);
    } while (value != 0);
    
    return encoded;
}

// Decodes a ULEB128 byte stream back into a uint32_t
uint32_t decodeULEB128(const std::vector<uint8_t>& bytes) {
    uint32_t result = 0;
    uint32_t shift = 0;
    
    for (uint8_t byte : bytes) {
        uint32_t slice = byte & 0x7F; // Strip the continuation bit
        result |= (slice << shift);   // Shift payload back into original position
        shift += 7;
        
        if ((byte & 0x80) == 0) {     // Continuation Bit is 0 -> End of integer
            break;
        }
    }
    return result;
}

int main() {
    uint32_t originalNumber = 214577;
    std::vector<uint8_t> leb128Memory = encodeULEB128(originalNumber);

    std::cout << "Original Value: " << originalNumber << "\n";
    std::cout << "Standard 32-bit Storage Size : " << sizeof(originalNumber) << " bytes\n";
    std::cout << "LEB128 Storage Size          : " << leb128Memory.size() << " bytes\n\n";

    std::cout << "=== PHYSICAL MEMORY LAYOUT ===\n";
    std::cout << "Address      | Offset | Raw Hex | Binary     | Cont. Bit | 7-Bit Payload\n";
    std::cout << "------------------------------------------------------------------------\n";

    const uint8_t* rawMemoryPtr = leb128Memory.data(); // Pointer to contiguous RAM buffer

    for (size_t i = 0; i < leb128Memory.size(); ++i) {
        uint8_t byte = rawMemoryPtr[i];
        const void* address = static_cast<const void*>(rawMemoryPtr + i);
        
        bool continuationBit = (byte & 0x80) != 0;
        uint8_t payload = byte & 0x7F;

        std::cout << address << " | +" << i << "     | 0x" 
                  << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(byte)
                  << "   | " << std::bitset<8>(byte)
                  << " | " << continuationBit
                  << "         | " << std::dec << static_cast<int>(payload) 
                  << " (0x" << std::hex << static_cast<int>(payload) << ")\n";
    }

    uint32_t reconstructed = decodeULEB128(leb128Memory);
    std::cout << "\nReconstructed Value from LEB128 Memory: " << std::dec << reconstructed << "\n";

    return 0;
}