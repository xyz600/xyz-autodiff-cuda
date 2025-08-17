# Claude Development Notes

## Build Commands

### Clean Build
```bash
task clean
```

### Debug Build
```bash
task build:debug
```

### Release Build
```bash
task build:release
```

## Testing

### Run All Tests
```bash
task test
```

### Run Specific Tests
```bash
# Variable tests
task build:debug && cd build/debug && ./tests/test_variable

# DenseMatrix tests  
task build:debug && cd build/debug && ./tests/test_dense_matrix
```
